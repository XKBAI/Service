// Chat management module for AI Services Gateway
class ChatManager {
    constructor(apiClient) {
        // Prevent multiple instances
        if (window.chatManagerInstance) {
            console.warn('ChatManager instance already exists, returning existing instance');
            return window.chatManagerInstance;
        }
        
        this.apiClient = apiClient;
        this.currentUserId = null; // Will be set after elements are initialized
        this.currentChatId = null;
        this.chatHistory = [];
        this.messages = [];
        this.isProcessing = false;
        this.uploadedFile = null;
        this.currentAudio = null; // Track current playing audio
        this.audioButtons = new Map(); // Track audio button states
        this.mathJaxRenderTimeout = null; // For debounced MathJax rendering
        this.eventListenersInitialized = false; // Prevent duplicate event listeners
        this.abortController = null; // For canceling ongoing requests
        this.lastSentMessage = null; // Track last sent message for duplicate detection
        this.lastSentTime = 0; // Track last sent time for duplicate detection
        
        // STT recording properties
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordingTimeout = null;

        this.initElements();
        this.initEventListeners();
        this.initCurrentUser();
        this.initSidebarState();
        this.initChatTypeSelector();
        this.initThinkingContainerHandlers();
        
        // Store instance globally to prevent duplicates
        window.chatManagerInstance = this;
    }

    initElements() {
        // Chat elements
        this.chatList = document.getElementById('chatList');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.attachBtn = document.getElementById('attachBtn');
        this.sttBtn = document.getElementById('sttBtn');
        this.fileInput = document.getElementById('fileInput');
        this.filePreview = document.getElementById('filePreview');
        this.currentChatTitle = document.getElementById('currentChatTitle');
        
        // User selector
        this.userSelector = document.getElementById('userUID');
        
        // Buttons
        this.newChatBtn = document.getElementById('newChatBtn');
        this.toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
        
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    initEventListeners() {
        // Prevent duplicate event listeners
        if (this.eventListenersInitialized) {
            return;
        }
        
        // Send message
        this.sendBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Send button clicked');
            this.sendMessage();
        });
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter key pressed');
                this.sendMessage();
            }
        });

        // Stop generation
        this.stopBtn.addEventListener('click', () => this.stopGeneration());

        // Auto-resize textarea
        this.messageInput.addEventListener('input', this.autoResizeTextarea.bind(this));

        // File upload
        this.attachBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', this.handleFileUpload.bind(this));
        
        // STT recording
        this.sttBtn.addEventListener('click', this.handleSTTClick.bind(this));

        // User selector
        this.userSelector.addEventListener('change', this.handleUserChange.bind(this));

        // Chat type selector
        this.chatTypeSelector = document.getElementById('chatType');
        if (this.chatTypeSelector) {
            this.chatTypeSelector.addEventListener('change', this.handleChatTypeChange.bind(this));
        }

        // New chat
        this.newChatBtn.addEventListener('click', () => this.createNewChat());

        // Toggle sidebar
        this.toggleSidebarBtn.addEventListener('click', () => this.toggleSidebar());
        
        // Resize handles
        this.initResizeHandles();
        
        // Mark as initialized
        this.eventListenersInitialized = true;
    }

    // Initialize current user from selector
    initCurrentUser() {
        // Ensure userSelector exists and has options
        if (this.userSelector) {
            // Wait for options to be populated
            if (this.userSelector.options.length > 0) {
                // Ensure a user is selected
                if (!this.userSelector.value && this.userSelector.options.length > 0) {
                    this.userSelector.selectedIndex = 0;
                }
                this.currentUserId = this.userSelector.value || this.userSelector.options[0].value;
                console.log('Initialized current user:', this.currentUserId);
            } else {
                // No options yet, use default and set up observer
                this.currentUserId = 'user_001';
                console.log('Initialized current user (default):', this.currentUserId);
                this.waitForUserOptions();
            }
        } else {
            this.currentUserId = 'user_001'; // Default fallback
            console.log('Initialized current user (fallback):', this.currentUserId);
        }
    }

    // Wait for user options to be populated
    waitForUserOptions() {
        if (!this.userSelector) return;
        
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && this.userSelector.options.length > 0) {
                    console.log('User options populated, updating current user');
                    if (!this.userSelector.value) {
                        this.userSelector.selectedIndex = 0;
                    }
                    const newUserId = this.userSelector.value || this.userSelector.options[0].value;
                    
                    // Only update if the user ID actually changed
                    if (newUserId !== this.currentUserId) {
                        this.currentUserId = newUserId;
                        console.log('Updated current user to:', this.currentUserId);
                        
                        // Reload chats for the new user
                        this.loadUserChats().catch(console.error);
                    } else {
                        console.log('User ID unchanged, skipping reload');
                    }
                    
                    observer.disconnect();
                }
            });
        });
        
        observer.observe(this.userSelector, { childList: true });
        
        // Auto-disconnect after 5 seconds to prevent memory leaks
        setTimeout(() => observer.disconnect(), 5000);
    }

    // Initialize sidebar state from localStorage
    initSidebarState() {
        const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
        if (sidebarCollapsed) {
            this.setSidebarCollapsed(true, false); // Don't save to localStorage again
        }
    }

    // Initialize chat type selector from localStorage
    initChatTypeSelector() {
        if (this.chatTypeSelector) {
            const savedChatType = localStorage.getItem('selectedChatType');
            if (savedChatType && this.chatTypeSelector.querySelector(`option[value="${savedChatType}"]`)) {
                this.chatTypeSelector.value = savedChatType;
            }
        }
    }

    // Initialize thinking container click handlers using event delegation
    initThinkingContainerHandlers() {
        if (this.messagesContainer) {
            // Use event delegation to handle clicks on thinking headers
            this.messagesContainer.addEventListener('click', (e) => {
                // Check if the clicked element is a thinking header or its child
                const thinkingHeader = e.target.closest('.thinking-header');
                if (thinkingHeader) {
                    const thinkingContainer = thinkingHeader.closest('.thinking-container');
                    if (thinkingContainer) {
                        thinkingContainer.classList.toggle('collapsed');
                        e.preventDefault();
                        e.stopPropagation();
                    }
                }
            });
        }
    }

    // Toggle sidebar collapse state
    toggleSidebar() {
        const isCurrentlyCollapsed = this.sidebar.classList.contains('collapsed');
        this.setSidebarCollapsed(!isCurrentlyCollapsed, true);
    }

    // Set sidebar collapsed state
    setSidebarCollapsed(collapsed, save = true) {
        if (collapsed) {
            this.sidebar.classList.add('collapsed');
            this.toggleSidebarBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
            this.toggleSidebarBtn.title = '展开侧边栏';
        } else {
            this.sidebar.classList.remove('collapsed');
            this.toggleSidebarBtn.innerHTML = '<i class="fas fa-bars"></i>';
            this.toggleSidebarBtn.title = '折叠侧边栏';
        }

        if (save) {
            localStorage.setItem('sidebarCollapsed', collapsed.toString());
        }
    }

    // Initialize resize handles for draggable functionality
    initResizeHandles() {
        // Sidebar width resize handle
        const sidebarResizeHandle = document.getElementById('sidebarResizeHandle');
        if (sidebarResizeHandle) {
            this.initSidebarResize(sidebarResizeHandle);
        }
        
        // Input area height resize handle
        const inputResizeHandle = document.getElementById('inputResizeHandle');
        if (inputResizeHandle) {
            this.initInputAreaResize(inputResizeHandle);
        }
    }

    // Initialize sidebar width resizing
    initSidebarResize(handle) {
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;

        const startResize = (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = parseInt(document.defaultView.getComputedStyle(this.sidebar).width, 10);
            document.addEventListener('mousemove', doResize);
            document.addEventListener('mouseup', stopResize);
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        };

        const doResize = (e) => {
            if (!isResizing) return;
            
            const deltaX = e.clientX - startX;
            const newWidth = startWidth + deltaX;
            
            // Constrain width between min and max values
            const minWidth = 200;
            const maxWidth = 500;
            const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
            
            this.sidebar.style.width = constrainedWidth + 'px';
            
            // Save to localStorage
            localStorage.setItem('sidebarWidth', constrainedWidth.toString());
        };

        const stopResize = () => {
            isResizing = false;
            document.removeEventListener('mousemove', doResize);
            document.removeEventListener('mouseup', stopResize);
            document.body.style.cursor = '';
        };

        handle.addEventListener('mousedown', startResize);
        
        // Load saved width
        const savedWidth = localStorage.getItem('sidebarWidth');
        if (savedWidth) {
            this.sidebar.style.width = savedWidth + 'px';
        }
    }

    // Initialize input area height resizing
    initInputAreaResize(handle) {
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        const inputArea = document.getElementById('inputArea');

        const startResize = (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = parseInt(document.defaultView.getComputedStyle(inputArea).height, 10);
            document.addEventListener('mousemove', doResize);
            document.addEventListener('mouseup', stopResize);
            document.body.style.cursor = 'row-resize';
            e.preventDefault();
        };

        const doResize = (e) => {
            if (!isResizing) return;
            
            const deltaY = startY - e.clientY; // Reverse direction for intuitive resizing
            const newHeight = startHeight + deltaY;
            
            // Constrain height between min and max values
            const minHeight = 120;
            const maxHeight = 400;
            const constrainedHeight = Math.max(minHeight, Math.min(maxHeight, newHeight));
            
            inputArea.style.height = constrainedHeight + 'px';
            
            // Save to localStorage
            localStorage.setItem('inputAreaHeight', constrainedHeight.toString());
        };

        const stopResize = () => {
            isResizing = false;
            document.removeEventListener('mousemove', doResize);
            document.removeEventListener('mouseup', stopResize);
            document.body.style.cursor = '';
        };

        handle.addEventListener('mousedown', startResize);
        
        // Load saved height
        const savedHeight = localStorage.getItem('inputAreaHeight');
        if (savedHeight) {
            inputArea.style.height = savedHeight + 'px';
        }
    }

    // Initialize chat manager
    async init() {
        try {
            // User ID is already initialized in constructor, just load chats
            await this.loadUserChats();
            
            // Only create new chat if no existing chats, don't auto-create on page refresh
            if (this.chatHistory.length === 0) {
                console.log('No existing chats found, showing empty state');
                this.showEmptyState();
            } else {
                console.log('Found existing chats, showing empty state until user selects one');
                this.showEmptyState();
            }
        } catch (error) {
            console.error('Failed to initialize chat manager:', error);
            Utils.showToast('初始化聊天失败', 'error');
        }
    }

    // Show empty state without auto-creating chat
    showEmptyState() {
        this.currentChatId = null;
        this.messages = [];
        this.currentChatTitle.textContent = '选择聊天或创建新聊天';
        this.renderMessages();
        this.clearFileUpload();
        
        // Focus on message input for better UX
        if (this.messageInput) {
            this.messageInput.focus();
        }
    }

    // Load user chat history
    async loadUserChats() {
        try {
            console.log('Loading chats for user:', this.currentUserId);
            if (!this.currentUserId || this.currentUserId === 'undefined') {
                console.warn('Invalid user ID, using default');
                this.currentUserId = 'user_001';
            }
            this.chatHistory = await this.apiClient.getUserChats(this.currentUserId);
            this.renderChatList();
        } catch (error) {
            console.error('Failed to load user chats for user:', this.currentUserId, error);
            this.chatHistory = [];
            this.renderChatList();
        }
    }

    // Render chat list in sidebar
    renderChatList() {
        if (!this.chatList) return;

        this.chatList.innerHTML = '';

        if (this.chatHistory.length === 0) {
            this.chatList.innerHTML = '<div class="no-chats">暂无聊天记录</div>';
            return;
        }

        this.chatHistory.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.dataset.chatId = chat.chat_session_id;
            
            if (chat.chat_session_id === this.currentChatId) {
                chatItem.classList.add('active');
            }

            chatItem.innerHTML = `
                <div class="chat-item-content">
                    <div class="chat-item-title">${Utils.truncateText(chat.chat_title)}</div>
                    <div class="chat-item-time">${Utils.formatTime(chat.chat_time)}</div>
                </div>
                <button class="chat-item-delete" title="删除聊天">
                    <i class="fas fa-trash"></i>
                </button>
            `;

            // Add click listener for chat content (not the delete button)
            const chatContent = chatItem.querySelector('.chat-item-content');
            chatContent.addEventListener('click', () => this.loadChat(chat.chat_session_id));
            
            // Add click listener for delete button
            const deleteBtn = chatItem.querySelector('.chat-item-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering chat load
                this.confirmDeleteChat(chat.chat_session_id, chat.chat_title);
            });
            this.chatList.appendChild(chatItem);
        });
    }
    
    // Confirm and delete chat
    async confirmDeleteChat(chatSessionId, chatTitle) {
        const confirmed = confirm(`确定要删除聊天"${chatTitle}"吗？此操作无法撤销。`);
        if (!confirmed) return;
        
        try {
            // Delete from backend
            await this.apiClient.deleteChat(chatSessionId);
            
            // If we're currently viewing this chat, clear the view
            if (this.currentChatId === chatSessionId) {
                this.currentChatId = null;
                this.messages = [];
                this.currentChatTitle.textContent = '请选择聊天或创建新聊天';
                this.renderMessages();
                this.clearFileUpload();
            }
            
            // Reload chat list
            await this.loadUserChats();
            
            Utils.showToast('聊天已删除', 'success');
        } catch (error) {
            console.error('Failed to delete chat:', error);
            Utils.showToast('删除聊天失败', 'error');
        }
    }

    // Load a specific chat
    async loadChat(chatId) {
        try {
            this.showLoading(true);
            
            const chatData = await this.apiClient.getChatBySessionId(chatId);
            
            this.currentChatId = chatId;
            this.messages = chatData.messages || [];
            this.currentChatTitle.textContent = chatData.chat_title || '未命名聊天';
            
            this.renderMessages();
            this.updateActiveChat();
            
            // Force scroll to bottom after loading chat content
            setTimeout(() => this.forceScrollToBottom(), 150);
            
        } catch (error) {
            console.error('Failed to load chat:', error);
            Utils.showToast('加载聊天失败', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // Create new chat immediately on backend with retry mechanism
    async createNewChat(maxRetries = 3) {
        let lastError;
        
        // Reset current state first
        this.currentChatId = null;
        this.messages = [];
        this.currentChatTitle.textContent = '新建聊天';
        this.renderMessages();
        this.clearFileUpload();
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                // Create chat on backend
                const now = new Date();
                const chatContent = {
                    chat_time: now.toLocaleDateString('sv-SE') + ' ' + now.toLocaleTimeString('sv-SE'),
                    messages: [], // Start with empty messages
                    chat_title: '新建聊天'
                };

                console.log(`Creating new chat on backend (attempt ${attempt})...`);
                const result = await this.apiClient.addChat(this.currentUserId, chatContent);
                
                if (!result || !result.chat_session_id) {
                    throw new Error('Backend returned invalid chat session ID');
                }
                
                this.currentChatId = result.chat_session_id;
                
                // Reload chat list to show the new chat
                await this.loadUserChats();
                this.updateActiveChat();
                
                // Focus on message input
                if (this.messageInput) {
                    this.messageInput.focus();
                }
                
                console.log('Created new chat with ID:', this.currentChatId);
                Utils.showToast('已创建新聊天', 'success');
                return; // Success
                
            } catch (error) {
                lastError = error;
                console.warn(`Create chat attempt ${attempt} failed:`, error.message);
                
                if (attempt < maxRetries) {
                    // Wait before retry
                    const delay = Math.min(1000 * attempt, 3000);
                    console.log(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                } else {
                    // All attempts failed
                    console.error('All create chat attempts failed:', lastError);
                    Utils.showToast(`创建聊天失败: ${lastError.message}`, 'error');
                    
                    // Ensure we still have a working UI state even if backend fails
                    this.currentChatId = null;
                    this.updateActiveChat();
                    
                    if (this.messageInput) {
                        this.messageInput.focus();
                    }
                    
                    // Don't throw error to allow UI to continue working
                    // throw lastError;
                }
            }
        }
    }

    // Update active chat in sidebar
    updateActiveChat() {
        const chatItems = this.chatList.querySelectorAll('.chat-item');
        chatItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.chatId === this.currentChatId) {
                item.classList.add('active');
            }
        });
    }

    // Handle user change
    async handleUserChange() {
        const newUserId = this.userSelector.value;
        if (newUserId && newUserId !== this.currentUserId) {
            console.log('User changed from', this.currentUserId, 'to', newUserId);
            
            // Prevent processing during user change
            if (this.isProcessing) {
                console.log('Cannot change user while processing message');
                this.userSelector.value = this.currentUserId; // Revert selection
                return;
            }
            
            this.currentUserId = newUserId;
            
            // Clean up any ongoing recording
            this.cleanupSTT();
            
            // Load user's existing chats
            await this.loadUserChats();
            
            // Load the most recent chat if exists, otherwise show empty state
            if (this.chatHistory.length > 0) {
                // Load the most recent chat (first in the list)
                console.log('Loading most recent chat:', this.chatHistory[0].chat_session_id);
                await this.loadChat(this.chatHistory[0].chat_session_id);
            } else {
                // Show empty state, don't auto-create chat
                console.log('No existing chats found, showing empty state');
                this.showEmptyState();
            }
            
            Utils.showToast(`已切换到用户: ${newUserId}`, 'info');
        } else if (!newUserId) {
            console.warn('Empty user ID selected');
            this.currentUserId = 'user_001'; // Fallback
        }
    }

    // Handle chat type change
    handleChatTypeChange() {
        const selectedType = this.chatTypeSelector.value;
        console.log('Chat type changed to:', selectedType);
        
        // You can add logic here to modify how requests are sent based on chat type
        // For example, adding different prompts or using different API endpoints
        
        // Save the selected chat type to localStorage for persistence
        localStorage.setItem('selectedChatType', selectedType);
        
        // Show toast to confirm the change
        const typeNames = {
            'base': '对话',
            'rag': '知识库',
            'deepresearch': '深度学习'
        };
        
        Utils.showToast(`已切换到${typeNames[selectedType]}模式`, 'info');
    }

    // Send message
    async sendMessage() {
        const timestamp = Date.now();
        console.log('sendMessage called at:', timestamp, 'isProcessing:', this.isProcessing);
        
        const messageText = this.messageInput.value.trim();
        
        if (!messageText && !this.uploadedFile) {
            console.log('No message text or file, returning');
            return;
        }

        if (this.isProcessing) {
            console.log('Message already being processed, ignoring duplicate call. Current isProcessing:', this.isProcessing);
            console.log('Send button disabled:', this.sendBtn.disabled);
            console.log('Input disabled:', this.messageInput.disabled);
            return;
        }
        
        // Additional protection: check if we just sent the same message
        if (this.lastSentMessage === messageText && timestamp - (this.lastSentTime || 0) < 2000) {
            console.log('Duplicate message detected within 2 seconds, ignoring');
            return;
        }
        
        this.lastSentMessage = messageText;
        this.lastSentTime = timestamp;
        
        try {
            // Set processing state and disable input immediately
            this.isProcessing = true;
            this.setInputDisabled(true);
            console.log('Starting message send process...', 'Text:', messageText, 'Chat ID:', this.currentChatId);

            let finalMessage = messageText;
            let fileContent = '';

            // Add user message to UI first (non-blocking) - but check for duplicates
            let userMessageElement;
            
            // Check if we already have this exact message (anti-duplicate protection)
            const now = Date.now();
            const existingUserMsg = this.messages.find(msg => 
                msg.role === 'user' && 
                msg.content === messageText && 
                now - new Date(msg.timestamp).getTime() < 5000 // within 5 seconds
            );
            
            if (existingUserMsg) {
                console.warn('Duplicate user message detected, skipping...', 
                    'Message:', messageText, 
                    'Existing timestamp:', existingUserMsg.timestamp,
                    'Time diff:', now - new Date(existingUserMsg.timestamp).getTime());
                this.isProcessing = false;
                this.setInputDisabled(false);
                return;
            }
            
            userMessageElement = this.addMessage('user', messageText, this.uploadedFile);
            
            // Process uploaded file if any (async, non-blocking)
            if (this.uploadedFile) {
                // Show file processing indicator in the message
                this.showFileProcessingIndicator(userMessageElement, this.uploadedFile.name);
                
                try {
                    const fileResult = await this.apiClient.processFileForChat(this.uploadedFile);
                    
                    if (fileResult.content) {
                        fileContent = `\n\n[文件内容: ${fileResult.filename} (${fileResult.fileSize})]\n${fileResult.content}`;
                        finalMessage = messageText + fileContent;
                        
                        // Update the user message with processed file content
                        this.updateMessageWithFileContent(userMessageElement, finalMessage);
                        Utils.showToast('文件处理完成', 'success');
                    } else {
                        Utils.showToast('文件处理完成，但未能识别内容', 'warning');
                    }
                    
                    this.hideFileProcessingIndicator(userMessageElement);
                } catch (error) {
                    console.error('File processing error:', error);
                    Utils.showToast(`文件处理失败: ${error.message}`, 'error');
                    this.hideFileProcessingIndicator(userMessageElement);
                    return;
                }
            }

            this.clearInput();
            this.clearFileUpload();

            // Ensure we have a chat session before proceeding
            if (!this.currentChatId) {
                console.log('No chat session, creating new one...');
                await this.createNewChat();
                if (!this.currentChatId) {
                    throw new Error('无法创建聊天会话，请稍后重试');
                }
            }

            // Save user message immediately with retry (only once)
            console.log('Saving user message to chat...');
            await this.saveChatWithRetry();

            // Show assistant typing indicator
            const typingId = this.addTypingIndicator();
            
            // Show stop button and hide send button
            this.showStopButton(true);

            try {
                // Create abort controller for this request
                this.abortController = new AbortController();
                
                // Send to LLM - use original message text as prompt, conversation history for context
                // Get history messages excluding the current user message and any streaming assistant messages
                const historyMessages = this.messages
                    .filter(msg => !(msg.role === 'user' && msg.content === messageText && 
                                   Date.now() - new Date(msg.timestamp).getTime() < 10000)) // Exclude current user message
                    .filter(msg => !(msg.role === 'assistant' && msg.isStreaming)) // Exclude streaming assistant messages
                    .map(msg => ({
                        role: msg.role,
                        content: msg.content
                    }));
                
                console.log('Sending to LLM - Message:', messageText);
                console.log('History messages count:', historyMessages.length);
                console.log('History messages:', historyMessages);
                
                // Get current chat type
                const currentChatType = this.chatTypeSelector ? this.chatTypeSelector.value : 'base';
                console.log('Using chat type:', currentChatType);
                
                const response = await this.apiClient.chatCompletion(
                    messageText, // Use original user input as main prompt
                    historyMessages, // Only historical messages, not including current user input
                    this.abortController.signal, // Pass abort signal
                    currentChatType // Pass selected chat type
                );
                
                console.log('LLM response received:', response);

                // Remove typing indicator
                this.removeTypingIndicator(typingId);

                // Handle streaming response
                const streamResult = await this.handleStreamingResponse(response);

                // Update assistant message content after streaming
                if (streamResult && streamResult.messageElement) {
                    // Use the raw text content from streaming, not HTML
                    const finalContent = streamResult.content.chat || '';
                    
                    // Find the assistant message we just created and update its content
                    const lastAssistantIndex = this.messages.length - 1;
                    console.log('Updating assistant message, index:', lastAssistantIndex, 'total messages:', this.messages.length);
                    
                    if (lastAssistantIndex >= 0 && this.messages[lastAssistantIndex].role === 'assistant') {
                        console.log('Found assistant message to update, current content length:', this.messages[lastAssistantIndex].content.length);
                        this.messages[lastAssistantIndex].content = finalContent; // Save raw text, not HTML
                        this.messages[lastAssistantIndex].isStreaming = false; // Mark as completed
                        console.log('Updated assistant message content, new length:', finalContent.length);
                    } else {
                        console.warn('No assistant message found to update!');
                    }
                    
                    // Do final formatting of the complete content
                    console.log('Doing final formatting of completed message');
                    console.log('Final content for formatting:', finalContent);
                    const chatSection = streamResult.messageElement.querySelector('.stream-section.chat-section') || 
                                       streamResult.messageElement.querySelector('.chat-section');
                    if (chatSection && finalContent) {
                        const formattedContent = this.formatMessageContent(finalContent);
                        console.log('Formatted content:', formattedContent);
                        chatSection.innerHTML = formattedContent;
                        // Clear the raw content attribute since we're doing final formatting
                        chatSection.removeAttribute('data-raw-content');
                    }
                    
                    // Enable user interaction immediately after content is displayed
                    this.isProcessing = false;
                    this.setInputDisabled(false);
                    this.showStopButton(false);
                    
                    // Save chat in background (non-blocking) after getting assistant response
                    this.saveChatWithRetry().then(() => {
                        console.log('Chat saved successfully after assistant response');
                    }).catch(error => {
                        console.warn('Failed to save chat after response:', error);
                    });
                    
                    // Final MathJax render for the completed message (async, non-blocking)
                    setTimeout(() => {
                        this.rerenderMathJax(streamResult.messageElement.querySelector('.message-content'));
                    }, 0);
                } else {
                    console.warn('No valid stream result received');
                }
            } catch (llmError) {
                this.removeTypingIndicator(typingId);
                if (llmError.name === 'AbortError') {
                    console.log('Request was aborted by user');
                    Utils.showToast('已停止生成', 'info');
                } else {
                    throw llmError;
                }
            }

        } catch (error) {
            console.error('Send message error:', error);
            if (error.name === 'AbortError') {
                Utils.showToast('已停止生成', 'info');
            } else {
                Utils.showToast('发送消息失败', 'error');
            }
            this.removeTypingIndicator();
        } finally {
            // Clean up abort controller
            this.abortController = null;
            
            // Ensure UI is reset (in case of errors)
            if (this.isProcessing) {
                this.isProcessing = false;
                this.setInputDisabled(false);
                this.showStopButton(false);
            }
            this.showLoading(false);
        }
    }

    // Handle streaming response from LLM
    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let messageElement = null;
        let currentContent = {
            chat: '',
            citations: [],
            citationContent: [],
            pdf: null
        };
        let currentStage = '';
        let buffer = ''; // Buffer for incomplete JSON chunks

        try {
            while (true) {
                // Check if request was aborted
                if (this.abortController && this.abortController.signal.aborted) {
                    console.log('Stream reading aborted by user');
                    break;
                }
                
                const { done, value } = await reader.read();
                
                if (done) {
                    console.log('Stream reading completed');
                    // Process any remaining buffer content
                    if (buffer.trim()) {
                        this.tryParseBufferedContent(buffer, currentContent, messageElement);
                    }
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                console.log('Received raw chunk:', chunk);
                
                // Add chunk to buffer
                buffer += chunk;
                
                // Split by newlines to find complete lines
                const lines = buffer.split('\n');
                
                // Keep the last line in buffer if it doesn't end with newline
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    
                    console.log('Processing streaming line:', line);
                    
                    // Handle both plain JSON and SSE format
                    let lineData = line.trim();
                    if (lineData.startsWith('data: ')) {
                        lineData = lineData.substring(6); // Remove 'data: ' prefix
                    }
                    
                    if (lineData === '[DONE]') {
                        console.log('Stream completed with [DONE] marker');
                        break;
                    }
                    
                    // Enhanced JSON parsing with better error handling
                    const parseResult = this.safeParseJSON(lineData);
                    if (parseResult.success) {
                        const data = parseResult.data;
                        console.log('Successfully parsed JSON:', data);
                        
                        // Create message element on first content (any type containing 'chat')
                        if (!messageElement && data.type && data.type.includes('chat') && data.content) {
                            console.log('Creating new streaming message element for type:', data.type);
                            messageElement = this.createEmptyStreamingMessage();
                        }
                        
                        // Process stream data and immediately update UI for chat content
                        await this.processStreamData(data, currentContent, messageElement);
                    } else {
                        console.warn('JSON parsing failed for line:', lineData, 'Error:', parseResult.error);
                        // Try to handle as partial JSON or raw text
                        this.handleNonJSONContent(lineData, currentContent, messageElement);
                    }
                }
            }
        } catch (error) {
            // Don't log AbortError as an error - it's expected when user stops generation
            if (error.name === 'AbortError') {
                console.log('Streaming was aborted by user');
            } else {
                console.error('Streaming error:', error);
                if (!messageElement) {
                    this.addMessage('assistant', '抱歉，处理您的请求时出现了错误。');
                }
            }
        }

        return { 
            content: currentContent, 
            messageElement: messageElement 
        };
    }

    // Safe JSON parsing with better error handling
    safeParseJSON(jsonString) {
        try {
            // First try: direct parsing
            const data = JSON.parse(jsonString);
            return { success: true, data: data };
        } catch (e1) {
            // Second try: fix common issues like missing opening brace
            try {
                let fixedJson = jsonString.trim();
                
                // Handle incomplete JSON that starts with property (missing opening brace)
                if (fixedJson.startsWith('"') && !fixedJson.startsWith('{"')) {
                    fixedJson = '{' + fixedJson;
                }
                
                // Handle incomplete JSON that ends abruptly (missing closing brace)
                if (!fixedJson.endsWith('}') && fixedJson.includes(':')) {
                    fixedJson = fixedJson + '}';
                }
                
                const data = JSON.parse(fixedJson);
                console.log('Fixed and parsed JSON:', fixedJson);
                return { success: true, data: data };
            } catch (e2) {
                return { success: false, error: e2.message, originalError: e1.message };
            }
        }
    }

    // Handle non-JSON content during streaming
    handleNonJSONContent(content, currentContent, messageElement) {
        // If we have a message element and the content looks like text, append it
        if (messageElement && content && typeof content === 'string') {
            // Filter out malformed JSON fragments but keep actual text
            const cleanContent = content.replace(/^[{":,\s]*/, '').replace(/[}":,\s]*$/, '');
            if (cleanContent.length > 0 && !cleanContent.match(/^[{}\[\]":,\s]*$/)) {
                console.log('Appending non-JSON text content:', cleanContent);
                currentContent.chat += cleanContent;
                this.appendChunkToMessage(messageElement, cleanContent);
            }
        }
    }

    // Try to parse any remaining buffered content
    tryParseBufferedContent(buffer, currentContent, messageElement) {
        const parseResult = this.safeParseJSON(buffer);
        if (parseResult.success) {
            console.log('Parsed final buffer content:', parseResult.data);
            if (messageElement) {
                this.processStreamData(parseResult.data, currentContent, messageElement);
            }
        } else {
            // Handle as text if it contains meaningful content
            this.handleNonJSONContent(buffer, currentContent, messageElement);
        }
    }

    // Process individual stream data chunks
    async processStreamData(data, currentContent, messageElement) {
        const { type, content } = data;
        console.log(`Processing stream data - Type: ${type}, Content: ${content}`);
        
        // Handle different message types based on your reference pattern
        if (type && type.includes('chat') && content !== undefined) {
            // Handle real-time chat content streaming for any type containing 'chat'
            console.log('Adding chat content:', content);
            currentContent.chat += String(content);
            // Real-time update: immediately append new chunk to UI
            if (messageElement && content) { // Only append non-empty content to UI
                console.log('Appending chunk to message element');
                this.appendChunkToMessage(messageElement, String(content));
            }
        } else if (type && type.includes('pdf') && content) {
            // Handle PDF content
            console.log('Processing PDF content:', content);
            console.log(content); // Log like your reference
            currentContent.pdf = content;
        } else {
            // Handle other types - log the entire data like your reference
            console.log('Other stream type:', data);
        }
        
        // Also handle specific cases
        switch (type) {
            case 'citation_content':
                // Handle citation content (images, references)
                if (Array.isArray(content)) {
                    currentContent.citationContent = currentContent.citationContent.concat(content);
                }
                break;
                
            case 'citation':
                // Handle citation data
                if (content && content.citations) {
                    currentContent.citations = currentContent.citations.concat(content.citations);
                }
                break;
                
            case 'pdf':
                // Handle PDF generation completion
                if (content && content.status === 'success') {
                    currentContent.pdf = content;
                }
                break;
        }
    }

    // Add message to UI and internal array
    addMessage(role, content, file = null) {
        const message = {
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            file: file
        };

        console.log('Adding message:', role, 'Content length:', content.length, 'Current messages count:', this.messages.length);
        this.messages.push(message);
        return this.renderMessage(message);
    }

    // Add streaming message with enhanced content
    addStreamingMessage(streamContent) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant streaming';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';
        
        this.updateStreamingContent(content, streamContent);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    // Create empty streaming message ready for real-time updates
    createEmptyStreamingMessage() {
        console.log('Creating empty streaming message, current messages count:', this.messages.length);
        
        // Add assistant message to messages array immediately
        const assistantMessage = {
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString(),
            isStreaming: true
        };
        this.messages.push(assistantMessage);
        console.log('Added assistant message to array, new count:', this.messages.length);

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant streaming';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    // Append new chunk to existing message in real-time
    appendChunkToMessage(messageElement, chunk) {
        console.log('appendChunkToMessage called with chunk:', chunk);
        
        const contentElement = messageElement.querySelector('.message-content');
        let chatSection = contentElement.querySelector('.stream-section.chat-section') || 
                         contentElement.querySelector('.chat-section');
        
        if (!chatSection) {
            // Create new chat section if it doesn't exist
            console.log('Creating new chat section');
            chatSection = document.createElement('div');
            chatSection.className = 'stream-section chat-section';
            contentElement.insertBefore(chatSection, contentElement.firstChild);
        }
        
        // For streaming, append raw text chunk and handle think tags properly
        const currentText = chatSection.getAttribute('data-raw-content') || '';
        const newText = currentText + chunk;
        console.log('Updating text from:', currentText, 'to:', newText);
        
        chatSection.setAttribute('data-raw-content', newText);
        
        // Process think tags and math in the accumulated text
        console.log('Processing accumulated text for think tags, length:', newText.length);
        
        // First decode any encoded math from backend
        let processedText = newText
            .replace(/_INLINE_MATH_([^_]+)INLINE_MATH/g, (match, encodedEq) => {
                try {
                    const equation = atob(encodedEq);
                    return `$${equation}$`;
                } catch (e) {
                    console.warn('Failed to decode inline math:', encodedEq);
                    return match;
                }
            })
            .replace(/_DISPLAY_MATH_([^_]+)DISPLAY_MATH/g, (match, encodedEq) => {
                try {
                    const equation = atob(encodedEq);
                    return `$$${equation}$$`;
                } catch (e) {
                    console.warn('Failed to decode display math:', encodedEq);
                    return match;
                }
            });
        
        // Check if we have complete thinking blocks to avoid breaking them during streaming
        const hasCompleteThink = processedText.includes('<think>') && processedText.includes('</think>');
        const hasIncompleteThink = processedText.includes('<think>') && !processedText.includes('</think>');
        
        if (hasCompleteThink || !hasIncompleteThink) {
            // Safe to update innerHTML when we have complete blocks or no incomplete blocks
            chatSection.innerHTML = this.formatMessageContentWithThinkTags(processedText);
        } else {
            // For incomplete thinking blocks, just append plain text to avoid breaking structure
            const existingContent = chatSection.innerHTML || '';
            const lastChunk = chunk.replace(/_INLINE_MATH_([^_]+)INLINE_MATH/g, (match, encodedEq) => {
                try {
                    const equation = atob(encodedEq);
                    return `$${equation}$`;
                } catch (e) {
                    return match;
                }
            });
            chatSection.innerHTML = existingContent + lastChunk;
        }
        
        // Trigger MathJax re-rendering for this specific element during streaming
        this.rerenderMathJax(chatSection);
        
        this.scrollToBottom();
    }

    // Format message content with proper think tag handling for streaming
    formatMessageContentWithThinkTags(content) {
        if (!content) return '';
        
        // For streaming, we need to handle incomplete think tags
        // First check if we have complete think blocks
        const completeThinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
        
        if (completeThinkMatch) {
            // We have complete think blocks, use the same extraction logic
            return this.formatMessageContent(content);
        } else if (content.includes('<think>')) {
            // We have an incomplete think block (streaming)
            const thinkIndex = content.indexOf('<think>');
            const beforeThink = content.substring(0, thinkIndex);
            const afterThink = content.substring(thinkIndex + '<think>'.length);
            
            let result = '';
            
            // Add any content before think tag as response
            if (beforeThink.trim()) {
                result += this.renderMarkdownContent(beforeThink.trim());
            }
            
            // Add thinking container with streaming content
            if (afterThink.trim()) {
                result += this.createThinkingContainer(afterThink.trim());
            }
            
            return result;
        } else {
            // No think tags, just render as normal content
            return this.renderMarkdownContent(content);
        }
    }

    // Create collapsible thinking container (always collapsed by default, user can expand)
    createThinkingContainer(content) {
        // Generate unique ID for the container
        const containerId = 'thinking-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        
        return `
            <div class="thinking-container collapsed" id="${containerId}">
                <div class="thinking-header">
                    <i class="fas fa-brain"></i>
                    <span>Thinking...</span>
                    <i class="fas fa-chevron-down thinking-arrow"></i>
                </div>
                <div class="thinking-content">
                    ${content}
                </div>
            </div>
        `;
    }

    // Basic markdown formatting without think tag processing
    basicMarkdownFormat(content) {
        if (!content) return '';
        
        try {
            // Configure marked with enhanced options for better rendering
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {}
                    }
                    return hljs.highlightAuto(code).value;
                },
                breaks: true, // Convert \n to <br>
                gfm: true, // GitHub Flavored Markdown
                tables: true, // Support tables
                pedantic: false, // Less strict, more forgiving
                sanitize: false, // Allow HTML (needed for math)
                smartLists: true, // Better list handling
                smartypants: true // Better typography (quotes, dashes, etc)
            });
            
            // First, handle already encoded math from backend
            let processedContent = content
                .replace(/_INLINE_MATH_([^_]+)INLINE_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$${equation}$`;
                    } catch (e) {
                        console.warn('Failed to decode inline math:', encodedEq);
                        return match;
                    }
                })
                .replace(/_DISPLAY_MATH_([^_]+)DISPLAY_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$$${equation}$$`;
                    } catch (e) {
                        console.warn('Failed to decode display math:', encodedEq);
                        return match;
                    }
                });
            
            // Then preserve LaTeX equations by temporarily replacing them
            processedContent = processedContent
                .replace(/\$\$([\s\S]*?)\$\$/g, (match, equation) => {
                    return `__DISPLAY_MATH_${btoa(equation)}_DISPLAY_MATH__`;
                })
                .replace(/\$([^$\n]+?)\$/g, (match, equation) => {
                    return `__INLINE_MATH_${btoa(equation)}_INLINE_MATH__`;
                })
                // Better code block handling
                .replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
                    return `\n\`\`\`${lang}\n${code.trim()}\n\`\`\`\n`;
                });
            
            // Render markdown to HTML
            let htmlContent = marked.parse(processedContent);
            
            // Restore LaTeX equations
            htmlContent = htmlContent
                .replace(/__DISPLAY_MATH_([^_]+)_DISPLAY_MATH__/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `<div class="math-display">$$${equation}$$</div>`;
                    } catch (e) {
                        return match;
                    }
                })
                .replace(/__INLINE_MATH_([^_]+)_INLINE_MATH__/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `<span class="math-inline">$${equation}$</span>`;
                    } catch (e) {
                        return match;
                    }
                });
            
            return htmlContent;
            
        } catch (error) {
            console.error('Markdown parsing error:', error);
            // Enhanced fallback with better formatting and think tag handling
            return content
                // First process think tags into collapsible containers
                .replace(/<think>([\s\S]*?)<\/think>/gi, (match, content) => this.createThinkingContainer(content))
                .replace(/<think>/gi, '<div class="thinking-container collapsed"><div class="thinking-header"><i class="fas fa-brain"></i> <span>Thinking...</span> <i class="fas fa-chevron-down thinking-arrow"></i></div><div class="thinking-content">')
                .replace(/<\/think>/gi, '</div></div>')
                // First handle encoded math
                .replace(/_INLINE_MATH_([^_]+)INLINE_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$${equation}$`;
                    } catch (e) {
                        console.warn('Failed to decode inline math:', encodedEq);
                        return match;
                    }
                })
                .replace(/_DISPLAY_MATH_([^_]+)DISPLAY_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$$${equation}$$`;
                    } catch (e) {
                        console.warn('Failed to decode display math:', encodedEq);
                        return match;
                    }
                })
                // Then do basic formatting
                .replace(/^---+$/gm, '<hr>')  // Horizontal rules
                .replace(/^#{6}\s+(.*$)/gm, '<h6>$1</h6>')  // h6
                .replace(/^#{5}\s+(.*$)/gm, '<h5>$1</h5>')  // h5
                .replace(/^#{4}\s+(.*$)/gm, '<h4>$1</h4>')  // h4
                .replace(/^#{3}\s+(.*$)/gm, '<h3>$1</h3>')  // h3
                .replace(/^#{2}\s+(.*$)/gm, '<h2>$1</h2>')  // h2
                .replace(/^#{1}\s+(.*$)/gm, '<h1>$1</h1>')  // h1
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`([^`]+)`/g, '<code>$1</code>');
        }
    }

    // Basic formatting for streaming content (without processing think tags)
    basicFormatContent(content) {
        if (!content) return '';
        
        // Only do basic markdown without think tag processing
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
    }

    // Update streaming message content
    updateStreamingMessage(messageElement, streamContent) {
        const contentElement = messageElement.querySelector('.message-content');
        this.updateStreamingContent(contentElement, streamContent);
        this.scrollToBottom();
    }

    // Update streaming content with all sections
    updateStreamingContent(contentElement, streamContent) {
        let html = '';
        
        // Main chat content
        if (streamContent.chat) {
            html += `<div class="stream-section chat-section">${this.formatMessageContent(streamContent.chat)}</div>`;
        }
        
        // Citation content (images/references)
        if (streamContent.citationContent && streamContent.citationContent.length > 0) {
            html += `<div class="stream-section citation-content-section">`;
            html += `<h4><i class="fas fa-images"></i> 引用内容</h4>`;
            html += `<div class="citation-content-grid">`;
            
            streamContent.citationContent.forEach(item => {
                html += `
                    <div class="citation-item">
                        <img src="${item.jpg_path}" alt="${item.title}" class="citation-image" loading="lazy">
                        <div class="citation-info">
                            <div class="citation-title">${item.title}</div>
                            ${item.url ? `<a href="${item.url}" target="_blank" class="citation-url">查看来源</a>` : ''}
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        // Citations list
        if (streamContent.citations && streamContent.citations.length > 0) {
            html += `<div class="stream-section citations-section">`;
            html += `<h4><i class="fas fa-quote-left"></i> 参考文献</h4>`;
            html += `<ol class="citations-list">`;
            
            streamContent.citations.forEach(citation => {
                Object.values(citation).forEach(value => {
                    html += `<li class="citation-item">${value}</li>`;
                });
            });
            
            html += `</ol></div>`;
        }
        
        // PDF download
        if (streamContent.pdf) {
            html += `
                <div class="stream-section pdf-section">
                    <h4><i class="fas fa-file-pdf"></i> PDF生成完成</h4>
                    <div class="pdf-download">
                        <i class="fas fa-download"></i>
                        <a href="${streamContent.pdf.download_url}" download="${streamContent.pdf.filename}" class="pdf-link">
                            下载 ${streamContent.pdf.filename}
                        </a>
                        <span class="file-size">(${streamContent.pdf.file_size})</span>
                    </div>
                </div>
            `;
        }
        
        // Add TTS button
        if (streamContent.chat) {
            html += `
                <div class="message-actions">
                    <button class="tts-btn" onclick="chatManager.playTTS('${streamContent.chat.replace(/'/g, "\\'").replace(/"/g, '\\"')}', this)">
                        <i class="fas fa-volume-up"></i> 转语音
                    </button>
                </div>
            `;
        }
        
        contentElement.innerHTML = html;
        
        // Add click handlers for citation images
        const citationImages = contentElement.querySelectorAll('.citation-image');
        citationImages.forEach(img => {
            img.addEventListener('click', () => this.showImageModal(img.src, img.alt));
        });
    }

    // Render a single message
    renderMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = message.role === 'user' ? '👤' : '🤖';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = this.formatMessageContent(message.content);

        // Add TTS button for assistant messages
        if (message.role === 'assistant') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'tts-btn';
            ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> 转语音';
            ttsBtn.addEventListener('click', (e) => this.playTTS(message.content, e.target));
            
            actions.appendChild(ttsBtn);
            content.appendChild(actions);
        }

        // Add file indicator for user messages with files
        if (message.role === 'user' && message.file) {
            const fileIndicator = document.createElement('div');
            fileIndicator.className = 'file-indicator';
            fileIndicator.innerHTML = `<i class="fas fa-paperclip"></i> ${message.file.name}`;
            content.insertBefore(fileIndicator, content.firstChild);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    // Update message content (for streaming)
    updateMessageContent(messageElement, newContent) {
        const contentElement = messageElement.querySelector('.message-content');
        contentElement.innerHTML = this.formatMessageContent(newContent);

        // Re-add TTS button if it's an assistant message
        if (messageElement.classList.contains('assistant')) {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'tts-btn';
            ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> 转语音';
            ttsBtn.addEventListener('click', (e) => this.playTTS(newContent, e.target));
            
            actions.appendChild(ttsBtn);
            contentElement.appendChild(actions);
        }

        this.scrollToBottom();
    }

    // Format message content with enhanced markdown and LaTeX support
    formatMessageContent(content) {
        if (!content) return '';
        
        // First extract thinking and response parts
        const { thinkingPart, responsePart } = this.extractThinkingAndResponse(content);
        
        let result = '';
        
        // Add thinking container if there's thinking content
        if (thinkingPart) {
            result += this.createThinkingContainer(thinkingPart);
        }
        
        // Add response part (if any)
        if (responsePart) {
            result += this.renderMarkdownContent(responsePart);
        }
        
        return result;
    }
    
    // Extract thinking and response parts from content
    extractThinkingAndResponse(content) {
        // Clean up any malformed markers first
        let cleanContent = content
            .replace(/[_]*THINK_BLOCK_[^_\s]*_THINK_BLOCK[_]*/g, '')
            .replace(/THINK_BLOCK_[^\s]*/g, '');
        
        // Check if there's a thinking part
        const thinkMatch = cleanContent.match(/<think>([\s\S]*?)<\/think>/i);
        
        let thinkingPart = '';
        let responsePart = '';
        
        if (thinkMatch) {
            // Extract thinking content
            thinkingPart = thinkMatch[1].trim();
            
            // Extract everything after </think> as response
            const thinkEndIndex = cleanContent.indexOf('</think>') + '</think>'.length;
            responsePart = cleanContent.substring(thinkEndIndex).trim();
        } else {
            // No thinking part, everything is response
            responsePart = cleanContent.trim();
        }
        
        return { thinkingPart, responsePart };
    }
    
    // Render markdown content (without think tags)
    renderMarkdownContent(content) {
        if (!content) return '';
        
        try {
            // Configure marked with enhanced options for better rendering
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {}
                    }
                    return hljs.highlightAuto(code).value;
                },
                breaks: true, // Convert \n to <br>
                gfm: true, // GitHub Flavored Markdown
                tables: true, // Support tables
                pedantic: false, // Less strict, more forgiving
                sanitize: false, // Allow HTML (needed for math)
                smartLists: true, // Better list handling
                smartypants: true // Better typography (quotes, dashes, etc)
            });
            
            // First, handle already encoded math from backend
            let processedContent = content
                .replace(/_INLINE_MATH_([^_]+)INLINE_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$${equation}$`;
                    } catch (e) {
                        console.warn('Failed to decode inline math:', encodedEq);
                        return match;
                    }
                })
                .replace(/_DISPLAY_MATH_([^_]+)DISPLAY_MATH/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `$$${equation}$$`;
                    } catch (e) {
                        console.warn('Failed to decode display math:', encodedEq);
                        return match;
                    }
                });
            
            // Then preserve LaTeX equations by temporarily replacing them
            processedContent = processedContent
                .replace(/\$\$([\s\S]*?)\$\$/g, (match, equation) => {
                    return `__DISPLAY_MATH_${btoa(equation)}_DISPLAY_MATH__`;
                })
                .replace(/\$([^$\n]+?)\$/g, (match, equation) => {
                    return `__INLINE_MATH_${btoa(equation)}_INLINE_MATH__`;
                })
                // Better code block handling
                .replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
                    return `\n\`\`\`${lang}\n${code.trim()}\n\`\`\`\n`;
                });
            
            // Render markdown to HTML
            let htmlContent = marked.parse(processedContent);
            
            // Restore LaTeX equations
            htmlContent = htmlContent
                .replace(/__DISPLAY_MATH_([^_]+)_DISPLAY_MATH__/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `<div class="math-display">$$${equation}$$</div>`;
                    } catch (e) {
                        return match;
                    }
                })
                .replace(/__INLINE_MATH_([^_]+)_INLINE_MATH__/g, (match, encodedEq) => {
                    try {
                        const equation = atob(encodedEq);
                        return `<span class="math-inline">$${equation}$</span>`;
                    } catch (e) {
                        return match;
                    }
                });
            
            return htmlContent;
            
        } catch (error) {
            console.error('Markdown parsing error:', error);
            // Enhanced fallback with better formatting
            return content
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                .replace(/^### (.*$)/gm, '<h3>$1</h3>');
        }
    }

    // Render all messages
    renderMessages() {
        if (!this.messagesContainer) return;

        // Clear container but keep welcome message if no messages
        this.messagesContainer.innerHTML = '';

        if (this.messages.length === 0) {
            const hasExistingChats = this.chatHistory && this.chatHistory.length > 0;
            const welcomeText = hasExistingChats 
                ? '请从左侧选择一个聊天，或点击"新建聊天"开始对话。'
                : '点击"新建聊天"开始与AI助手对话！支持上传PDF和图片文件进行分析。';
            
            this.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <i class="fas fa-robot"></i>
                    <h3>欢迎使用 学魁榜AI</h3>
                    <p>${welcomeText}</p>
                </div>
            `;
            return;
        }

        this.messages.forEach(message => {
            this.renderMessage(message);
        });
        
        // Force scroll to bottom when initially loading chat
        this.forceScrollToBottom();
    }

    // Add typing indicator
    addTypingIndicator() {
        const typingId = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = typingId;

        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        // Add typing animation styles if not present
        if (!document.getElementById('typing-styles')) {
            const style = document.createElement('style');
            style.id = 'typing-styles';
            style.textContent = `
                .typing-indicator {
                    display: flex;
                    gap: 4px;
                    align-items: center;
                }
                .typing-indicator span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #667eea;
                    animation: typing 1.4s infinite ease-in-out;
                }
                .typing-indicator span:nth-child(2) {
                    animation-delay: 0.2s;
                }
                .typing-indicator span:nth-child(3) {
                    animation-delay: 0.4s;
                }
                @keyframes typing {
                    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
                    30% { transform: translateY(-10px); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }

        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
        return typingId;
    }

    // Remove typing indicator
    removeTypingIndicator(typingId = null) {
        if (typingId) {
            const element = document.getElementById(typingId);
            if (element) element.remove();
        } else {
            // Remove all typing indicators
            const typingIndicators = this.messagesContainer.querySelectorAll('[id^="typing-"]');
            typingIndicators.forEach(el => el.remove());
        }
    }

    // Play TTS for message with pause/resume functionality
    async playTTS(text, buttonElement) {
        try {
            // Check if this is the same audio that is currently paused
            if (this.currentAudio && this.currentAudio.paused && 
                buttonElement.dataset.audioText === text) {
                // Resume the paused audio
                await this.currentAudio.play();
                this.updateTTSButtonState(buttonElement, true);
                Utils.showToast('继续播放语音', 'info');
                return;
            }
            
            // If audio is currently playing, pause it
            if (this.currentAudio && !this.currentAudio.paused) {
                this.currentAudio.pause();
                this.updateTTSButtonState(buttonElement, 'paused');
                Utils.showToast('语音已暂停', 'info');
                return;
            }

            // Stop any existing audio (different text)
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            // Update button to loading state
            this.updateTTSButtonState(buttonElement, 'loading');
            this.resetAllTTSButtons(buttonElement);

            const audioBlob = await this.apiClient.synthesizeText(text);
            const audioUrl = URL.createObjectURL(audioBlob);
            this.currentAudio = new Audio(audioUrl);
            
            // Ensure audio doesn't loop
            this.currentAudio.loop = false;
            
            // Store text reference for resume functionality
            buttonElement.dataset.audioText = text;
            
            this.currentAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
                this.updateTTSButtonState(buttonElement, false);
                buttonElement.removeAttribute('data-audio-text');
                Utils.showToast('语音播放完成', 'success');
            };
            
            this.currentAudio.onerror = () => {
                this.currentAudio = null;
                this.updateTTSButtonState(buttonElement, false);
                buttonElement.removeAttribute('data-audio-text');
                Utils.showToast('语音播放失败', 'error');
            };
            
            await this.currentAudio.play();
            this.updateTTSButtonState(buttonElement, true);
            Utils.showToast('正在播放语音', 'info');
            
        } catch (error) {
            console.error('TTS error:', error);
            this.updateTTSButtonState(buttonElement, false);
            buttonElement.removeAttribute('data-audio-text');
            Utils.showToast('语音合成失败', 'error');
        }
    }

    // Update TTS button state with visual feedback
    updateTTSButtonState(buttonElement, state) {
        if (!buttonElement) return;
        
        switch (state) {
            case 'loading':
                buttonElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 合成中...';
                buttonElement.disabled = true;
                buttonElement.classList.remove('playing', 'paused');
                break;
            case true: // playing
                buttonElement.innerHTML = '<i class="fas fa-pause"></i> 暂停';
                buttonElement.disabled = false;
                buttonElement.classList.add('playing');
                buttonElement.classList.remove('paused');
                break;
            case 'paused': // paused
                buttonElement.innerHTML = '<i class="fas fa-play"></i> 继续';
                buttonElement.disabled = false;
                buttonElement.classList.add('paused');
                buttonElement.classList.remove('playing');
                break;
            case false: // stopped
                buttonElement.innerHTML = '<i class="fas fa-volume-up"></i> 转语音';
                buttonElement.disabled = false;
                buttonElement.classList.remove('playing', 'paused');
                break;
        }
    }

    // Reset all TTS buttons except the active one
    resetAllTTSButtons(exceptButton = null) {
        const allTTSButtons = document.querySelectorAll('.tts-btn');
        allTTSButtons.forEach(btn => {
            if (btn !== exceptButton) {
                this.updateTTSButtonState(btn, false);
            }
        });
    }

    // Handle file upload
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const validation = Utils.validateFile(file);
        if (!validation.valid) {
            Utils.showToast(validation.error, 'error');
            return;
        }

        this.uploadedFile = file;
        this.showFilePreview(file);
        
        // Clear file input
        event.target.value = '';
    }

    // Show file preview
    showFilePreview(file) {
        const fileName = this.filePreview.querySelector('.file-name');
        const removeBtn = this.filePreview.querySelector('.remove-file-btn');
        
        fileName.textContent = `${file.name} (${Utils.formatFileSize(file.size)})`;
        this.filePreview.style.display = 'block';
        
        removeBtn.onclick = () => this.clearFileUpload();
    }

    // Clear file upload
    clearFileUpload() {
        this.uploadedFile = null;
        this.filePreview.style.display = 'none';
        this.fileInput.value = '';
    }

    // Save chat if needed
    async saveChatIfNeeded() {
        if (this.messages.length === 0) return;

        try {
            if (!this.currentChatId) {
                // This should not happen with the new logic, but keep as fallback
                console.warn('No chat ID found, creating new chat as fallback');
                await this.createNewChat();
                if (!this.currentChatId) {
                    throw new Error('无法创建聊天会话');
                }
            }

            // Update existing chat with new messages
            const now = new Date();
            const chatSession = {
                chat_session_id: this.currentChatId,
                chat_time: now.toLocaleDateString('sv-SE') + ' ' + now.toLocaleTimeString('sv-SE'),
                messages: this.messages.map(msg => ({
                    role: msg.role,
                    content: msg.content
                }))
            };

            console.log('Saving chat with', this.messages.length, 'messages to session', this.currentChatId);
            await this.apiClient.updateChat(chatSession);
            console.log('Chat saved successfully');

            // Generate and update title if this is the first real conversation
            if (this.messages.length >= 2 && this.currentChatTitle.textContent === '新建聊天') {
                try {
                    const chatTitle = await this.generateChatTitle();
                    if (chatTitle && chatTitle !== '新建聊天') {
                        this.currentChatTitle.textContent = chatTitle;
                        
                        // Update title on backend as well
                        await this.apiClient.updateChatTitle(this.currentChatId, chatTitle);
                        console.log('Chat title updated to:', chatTitle);
                    }
                } catch (titleError) {
                    console.warn('Failed to generate/update chat title:', titleError);
                    // Don't fail the entire save operation for title issues
                }
            }

            // Reload chat list to reflect any changes
            await this.loadUserChats();
            this.updateActiveChat();
        } catch (error) {
            console.error('Failed to save chat:', error);
            throw error; // Re-throw to allow caller to handle
        }
    }
    
    // Save chat with retry mechanism
    async saveChatWithRetry(maxRetries = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                await this.saveChatIfNeeded();
                return; // Success
            } catch (error) {
                lastError = error;
                console.warn(`Chat save attempt ${attempt} failed:`, error.message);
                
                if (attempt < maxRetries) {
                    // Wait before retry (exponential backoff)
                    const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
                    console.log(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                } else {
                    // Last attempt failed
                    console.error('All chat save attempts failed:', lastError);
                    Utils.showToast('保存聊天失败，请检查网络连接', 'error');
                    throw lastError;
                }
            }
        }
    }

    // Generate chat title based on conversation
    async generateChatTitle() {
        try {
            if (this.messages.length >= 2) {
                const titleResult = await this.apiClient.getChatTitle(
                    this.messages.slice(0, 4) // Use first few messages
                );
                return titleResult.title || '新建聊天';
            }
        } catch (error) {
            console.error('Failed to generate chat title:', error);
        }
        
        return '新建聊天';
    }

    // Utility methods
    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    clearInput() {
        this.messageInput.value = '';
        this.autoResizeTextarea();
    }

    setInputDisabled(disabled) {
        // Don't disable the input field itself - let users type during generation
        // this.messageInput.disabled = disabled;
        this.sendBtn.disabled = disabled;
        this.attachBtn.disabled = disabled;
        this.sttBtn.disabled = disabled;
        
        // Add visual indication when disabled
        if (disabled) {
            this.messageInput.setAttribute('data-processing', 'true');
            this.messageInput.placeholder = '正在生成回复中，您可以输入但暂时无法发送...';
        } else {
            this.messageInput.removeAttribute('data-processing');
            this.messageInput.placeholder = '输入消息...';
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            const container = this.messagesContainer;
            if (!container) return;
            
            // Check if user is near the bottom (within 100px)
            const isNearBottom = container.scrollTop + container.clientHeight >= 
                               container.scrollHeight - 100;
            
            // Only auto-scroll if user is already near the bottom
            if (isNearBottom) {
                container.scrollTop = container.scrollHeight;
            }
        }, 100);
    }
    
    // Force scroll to bottom (for manual scroll or initial load)
    forceScrollToBottom() {
        setTimeout(() => {
            if (this.messagesContainer) {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }
        }, 100);
    }

    // Stop ongoing generation and allow immediate sending
    stopGeneration() {
        if (this.abortController && !this.abortController.signal.aborted) {
            console.log('Stopping generation...');
            this.abortController.abort();
            
            // Update UI immediately
            this.removeTypingIndicator();
            this.showStopButton(false);
            this.isProcessing = false;
            this.setInputDisabled(false);
            
            // Focus back to input to allow immediate typing/sending
            if (this.messageInput) {
                this.messageInput.focus();
            }
            
            // Save the partial response if we have one
            setTimeout(() => {
                this.saveChatWithRetry().catch(error => {
                    console.warn('Failed to save chat after stopping:', error);
                });
            }, 100);
            
            Utils.showToast('已停止生成，现在可以发送新消息', 'info');
        }
    }

    // Show/hide stop button
    showStopButton(show) {
        if (this.stopBtn && this.sendBtn) {
            if (show) {
                this.stopBtn.style.display = 'inline-flex';
                this.sendBtn.style.display = 'none';
            } else {
                this.stopBtn.style.display = 'none';
                this.sendBtn.style.display = 'inline-flex';
            }
        }
    }

    showLoading(show, message = '处理中...') {
        if (show) {
            this.loadingOverlay.querySelector('p').textContent = message;
            this.loadingOverlay.style.display = 'flex';
        } else {
            this.loadingOverlay.style.display = 'none';
        }
    }

    // Show file processing indicator in message
    showFileProcessingIndicator(messageElement, fileName) {
        const contentElement = messageElement.querySelector('.message-content');
        const indicator = document.createElement('div');
        indicator.className = 'file-processing-indicator';
        indicator.innerHTML = `
            <div class="processing-spinner">
                <i class="fas fa-spinner fa-spin"></i>
            </div>
            <span class="processing-text">正在识别文件内容: ${fileName}...</span>
        `;
        contentElement.appendChild(indicator);
    }

    // Update file processing indicator status
    updateFileProcessingIndicator(messageElement, status) {
        const indicator = messageElement.querySelector('.file-processing-indicator');
        const textElement = indicator?.querySelector('.processing-text');
        if (textElement) {
            textElement.textContent = status;
        }
    }

    // Hide file processing indicator
    hideFileProcessingIndicator(messageElement) {
        const indicator = messageElement.querySelector('.file-processing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Update message with processed file content
    updateMessageWithFileContent(messageElement, newContent) {
        // Update UI
        const contentElement = messageElement.querySelector('.message-content');
        // Remove any existing content except the processing indicator
        const indicator = contentElement.querySelector('.file-processing-indicator');
        contentElement.innerHTML = this.formatMessageContent(newContent);
        if (indicator) {
            contentElement.appendChild(indicator);
        }

        // Update corresponding message in this.messages array
        // Find the most recent user message and update its content
        for (let i = this.messages.length - 1; i >= 0; i--) {
            if (this.messages[i].role === 'user') {
                this.messages[i].content = newContent;
                console.log('Updated user message content with file data');
                break;
            }
        }
    }

    // Show image in modal
    showImageModal(src, title) {
        // Create modal if it doesn't exist
        let modal = document.getElementById('imageModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'imageModal';
            modal.className = 'image-modal';
            modal.innerHTML = `
                <div class="image-modal-content">
                    <span class="image-modal-close">&times;</span>
                    <img class="image-modal-img" src="" alt="">
                    <div class="image-modal-caption"></div>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Add close handlers
            const closeBtn = modal.querySelector('.image-modal-close');
            closeBtn.onclick = () => modal.style.display = 'none';
            modal.onclick = (e) => {
                if (e.target === modal) modal.style.display = 'none';
            };
        }
        
        // Set image and show modal
        const img = modal.querySelector('.image-modal-img');
        const caption = modal.querySelector('.image-modal-caption');
        
        img.src = src;
        img.alt = title;
        caption.textContent = title;
        modal.style.display = 'block';
    }
    
    // STT (Speech-to-Text) functionality
    async handleSTTClick() {
        if (this.isRecording) {
            // Stop recording
            await this.stopRecording();
        } else {
            // Start recording
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            // Check browser compatibility
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('您的浏览器不支持录音功能');
            }
            
            if (!window.MediaRecorder) {
                throw new Error('您的浏览器不支持音频录制');
            }
            
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia(CONFIG.STT.AUDIO_CONSTRAINTS);
            
            // Set up MediaRecorder with best available format
            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/mp4';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = ''; // Use browser default
                    }
                }
            }
            
            this.mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
            
            this.audioChunks = [];
            this.isRecording = true;
            
            // Update button state
            this.updateSTTButtonState('recording');
            Utils.showToast('开始录音...', 'info');
            
            // Set up recording event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = async () => {
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
                
                if (this.audioChunks.length > 0) {
                    await this.processRecording();
                }
            };
            
            // Start recording
            this.mediaRecorder.start();
            
            // Set maximum recording time
            this.recordingTimeout = setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                    Utils.showToast('录音时间已达上限，自动停止', 'warning');
                }
            }, CONFIG.STT.MAX_RECORDING_TIME);
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.isRecording = false;
            this.updateSTTButtonState('idle');
            
            if (error.name === 'NotAllowedError') {
                Utils.showToast('需要麦克风权限才能进行语音输入', 'error');
            } else if (error.name === 'NotFoundError') {
                Utils.showToast('未找到可用的麦克风设备', 'error');
            } else {
                Utils.showToast('启动录音失败: ' + error.message, 'error');
            }
        }
    }
    
    async stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) {
            return;
        }
        
        this.isRecording = false;
        
        // Clear recording timeout
        if (this.recordingTimeout) {
            clearTimeout(this.recordingTimeout);
            this.recordingTimeout = null;
        }
        
        // Stop recording
        if (this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        this.updateSTTButtonState('processing');
        Utils.showToast('录音结束，正在转换文字...', 'info');
    }
    
    async processRecording() {
        try {
            // Create audio blob from chunks - use the same type that was recorded
            const audioBlob = new Blob(this.audioChunks, { 
                type: this.mediaRecorder.mimeType || 'audio/webm' 
            });
            console.log('Processing audio blob, size:', audioBlob.size, 'type:', audioBlob.type);
            
            if (audioBlob.size === 0) {
                throw new Error('录音文件为空');
            }
            
            // Convert to WAV format for better compatibility
            const wavBlob = await this.convertToWav(audioBlob);
            
            // Send to STT API
            const sttResult = await this.apiClient.transcribeAudio(wavBlob);
            
            if (sttResult && sttResult.text) {
                // Format the result and add to input
                this.handleSTTResult(sttResult.text);
                Utils.showToast('语音转换完成', 'success');
            } else {
                throw new Error('无法识别语音内容');
            }
            
        } catch (error) {
            console.error('STT processing error:', error);
            Utils.showToast('语音转换失败: ' + error.message, 'error');
        } finally {
            // Reset state
            this.audioChunks = [];
            this.updateSTTButtonState('idle');
        }
    }
    
    async convertToWav(webmBlob) {
        // For now, we'll send the webm blob directly
        // In a production environment, you might want to convert to WAV
        // This would require a more complex audio processing library
        return webmBlob;
    }
    
    handleSTTResult(transcribedText) {
        // Similar to OCR processing, format as a prompt
        const prompt = `[语音输入]: ${transcribedText}\n\n请基于以上语音内容进行回复。`;
        
        // Add to message input
        if (this.messageInput.value.trim()) {
            // If there's existing text, append with a newline
            this.messageInput.value += '\n\n' + prompt;
        } else {
            // If empty, just set the prompt
            this.messageInput.value = prompt;
        }
        
        // Auto-resize textarea
        this.autoResizeTextarea();
        
        // Focus on input
        this.messageInput.focus();
        
        console.log('STT result added to input:', transcribedText);
    }
    
    updateSTTButtonState(state) {
        if (!this.sttBtn) return;
        
        // Remove all state classes
        this.sttBtn.classList.remove('recording', 'processing');
        
        switch (state) {
            case 'recording':
                this.sttBtn.classList.add('recording');
                this.sttBtn.innerHTML = '<i class="fas fa-stop"></i>';
                this.sttBtn.title = '停止录音';
                break;
            case 'processing':
                this.sttBtn.classList.add('processing');
                this.sttBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                this.sttBtn.title = '正在处理...';
                break;
            case 'idle':
            default:
                this.sttBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                this.sttBtn.title = '语音输入';
                break;
        }
    }
    
    // Clean up STT resources
    cleanupSTT() {
        if (this.isRecording) {
            this.stopRecording();
        }
        
        if (this.recordingTimeout) {
            clearTimeout(this.recordingTimeout);
            this.recordingTimeout = null;
        }
        
        this.audioChunks = [];
        this.updateSTTButtonState('idle');
    }
    
    // Optimized MathJax re-rendering for specific elements
    rerenderMathJax(element = null) {
        if (!window.MathJax || !window.MathJax.typesetPromise) {
            return;
        }
        
        // Use a debounced approach to avoid excessive re-renders during rapid streaming
        if (this.mathJaxRenderTimeout) {
            clearTimeout(this.mathJaxRenderTimeout);
        }
        
        this.mathJaxRenderTimeout = setTimeout(() => {
            try {
                if (element) {
                    // Re-render only the specific element
                    window.MathJax.typesetPromise([element]).catch((err) => {
                        console.warn('MathJax element rendering error:', err);
                    });
                } else {
                    // Re-render the entire document (fallback)
                    window.MathJax.typesetPromise().catch((err) => {
                        console.warn('MathJax global rendering error:', err);
                    });
                }
            } catch (error) {
                console.warn('MathJax rendering failed:', error);
            }
        }, 200); // Debounce for 200ms
    }
}

// Export for global use
window.ChatManager = ChatManager;