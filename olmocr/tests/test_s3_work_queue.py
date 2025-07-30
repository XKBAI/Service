import asyncio
import datetime
import hashlib
import unittest
from typing import Dict, List
from unittest.mock import Mock, call, patch

from botocore.exceptions import ClientError

# Import the classes we're testing
from olmocr.work_queue import S3WorkQueue, WorkItem


class TestS3WorkQueue(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.s3_client = Mock()
        self.s3_client.exceptions.ClientError = ClientError
        self.work_queue = S3WorkQueue(self.s3_client, "s3://test-bucket/workspace")
        self.sample_paths = [
            "s3://test-bucket/data/file1.pdf",
            "s3://test-bucket/data/file2.pdf",
            "s3://test-bucket/data/file3.pdf",
        ]

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_compute_workgroup_hash(self):
        """Test hash computation is deterministic and correct"""
        paths = [
            "s3://test-bucket/data/file2.pdf",
            "s3://test-bucket/data/file1.pdf",
        ]

        # Hash should be the same regardless of order
        hash1 = S3WorkQueue._compute_workgroup_hash(paths)
        hash2 = S3WorkQueue._compute_workgroup_hash(reversed(paths))
        self.assertEqual(hash1, hash2)

    def test_init(self):
        """Test initialization of S3WorkQueue"""
        client = Mock()
        queue = S3WorkQueue(client, "s3://test-bucket/workspace/")

        self.assertEqual(queue.workspace_path, "s3://test-bucket/workspace")
        self.assertEqual(queue._index_path, "s3://test-bucket/workspace/work_index_list.csv.zstd")
        self.assertEqual(queue._output_glob, "s3://test-bucket/workspace/results/*.jsonl")

    def asyncSetUp(self):
        """Set up async test fixtures"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def asyncTearDown(self):
        """Clean up async test fixtures"""
        self.loop.close()

    def async_test(f):
        """Decorator for async test methods"""

        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(f(*args, **kwargs))
            finally:
                loop.close()

        return wrapper

    @async_test
    async def test_populate_queue_new_items(self):
        """Test populating queue with new items"""
        # Mock empty existing index
        with patch("olmocr.work_queue.download_zstd_csv", return_value=[]):
            with patch("olmocr.work_queue.upload_zstd_csv") as mock_upload:
                await self.work_queue.populate_queue(self.sample_paths, items_per_group=2)

                # Verify upload was called with correct data
                self.assertEqual(mock_upload.call_count, 1)
                _, _, lines = mock_upload.call_args[0]

                # Should create 2 work groups (2 files + 1 file)
                self.assertEqual(len(lines), 2)

                # Verify format of uploaded lines
                for line in lines:
                    parts = line.split(",")
                    self.assertGreaterEqual(len(parts), 2)  # Hash + at least one path
                    self.assertEqual(len(parts[0]), 40)  # SHA1 hash length

    @async_test
    async def test_populate_queue_existing_items(self):
        """Test populating queue with mix of new and existing items"""
        existing_paths = ["s3://test-bucket/data/existing1.pdf"]
        new_paths = ["s3://test-bucket/data/new1.pdf"]

        # Create existing index content
        existing_hash = S3WorkQueue._compute_workgroup_hash(existing_paths)
        existing_line = f"{existing_hash},{existing_paths[0]}"

        with patch("olmocr.work_queue.download_zstd_csv", return_value=[existing_line]):
            with patch("olmocr.work_queue.upload_zstd_csv") as mock_upload:
                await self.work_queue.populate_queue(existing_paths + new_paths, items_per_group=1)

                # Verify upload called with both existing and new items
                _, _, lines = mock_upload.call_args[0]
                self.assertEqual(len(lines), 2)
                self.assertIn(existing_line, lines)

    @async_test
    async def test_initialize_queue(self):
        """Test queue initialization"""
        # Mock work items and completed items
        work_paths = ["s3://test/file1.pdf", "s3://test/file2.pdf"]
        work_hash = S3WorkQueue._compute_workgroup_hash(work_paths)
        work_line = f"{work_hash},{work_paths[0]},{work_paths[1]}"

        completed_items = [f"s3://test-bucket/workspace/results/output_{work_hash}.jsonl"]

        with patch("olmocr.work_queue.download_zstd_csv", return_value=[work_line]):
            with patch("olmocr.work_queue.expand_s3_glob", return_value=completed_items):
                await self.work_queue.initialize_queue()

                # Queue should be empty since all work is completed
                self.assertTrue(self.work_queue._queue.empty())

    @async_test
    async def test_is_completed(self):
        """Test completed work check"""
        work_hash = "testhash123"

        # Test completed work
        self.s3_client.head_object.return_value = {"LastModified": datetime.datetime.now(datetime.timezone.utc)}
        self.assertTrue(await self.work_queue.is_completed(work_hash))

        # Test incomplete work
        self.s3_client.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        self.assertFalse(await self.work_queue.is_completed(work_hash))

    @async_test
    async def test_get_work(self):
        """Test getting work items"""
        # Setup test data
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Test getting available work
        self.s3_client.head_object.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        result = await self.work_queue.get_work()
        self.assertEqual(result, work_item)

        # Verify lock file was created
        self.s3_client.put_object.assert_called_once()
        bucket, key = self.s3_client.put_object.call_args[1]["Bucket"], self.s3_client.put_object.call_args[1]["Key"]
        self.assertTrue(key.endswith(f"output_{work_item.hash}.jsonl"))

    @async_test
    async def test_get_work_completed(self):
        """Test getting work that's already completed"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate completed work
        self.s3_client.head_object.return_value = {"LastModified": datetime.datetime.now(datetime.timezone.utc)}

        result = await self.work_queue.get_work()
        self.assertIsNone(result)  # Should skip completed work

    @async_test
    async def test_get_work_locked(self):
        """Test getting work that's locked by another worker"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate active lock
        recent_time = datetime.datetime.now(datetime.timezone.utc)
        self.s3_client.head_object.side_effect = [
            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # Not completed
            {"LastModified": recent_time},  # Active lock
        ]

        result = await self.work_queue.get_work()
        self.assertIsNone(result)  # Should skip locked work

    @async_test
    async def test_get_work_stale_lock(self):
        """Test getting work with a stale lock"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        # Simulate stale lock
        stale_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        self.s3_client.head_object.side_effect = [
            ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"),  # Not completed
            {"LastModified": stale_time},  # Stale lock
        ]

        result = await self.work_queue.get_work()
        self.assertEqual(result, work_item)  # Should take work with stale lock

    @async_test
    async def test_mark_done(self):
        """Test marking work as done"""
        work_item = WorkItem(hash="testhash123", work_paths=["s3://test/file1.pdf"])
        await self.work_queue._queue.put(work_item)

        await self.work_queue.mark_done(work_item)

        # Verify lock file was deleted
        self.s3_client.delete_object.assert_called_once()
        bucket, key = self.s3_client.delete_object.call_args[1]["Bucket"], self.s3_client.delete_object.call_args[1]["Key"]
        self.assertTrue(key.endswith(f"output_{work_item.hash}.jsonl"))

    def test_queue_size(self):
        """Test queue size property"""
        self.assertEqual(self.work_queue.size, 0)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self.work_queue._queue.put(WorkItem(hash="test1", work_paths=["path1"])))
        self.assertEqual(self.work_queue.size, 1)

        self.loop.run_until_complete(self.work_queue._queue.put(WorkItem(hash="test2", work_paths=["path2"])))
        self.assertEqual(self.work_queue.size, 2)

        self.loop.close()


if __name__ == "__main__":
    unittest.main()
