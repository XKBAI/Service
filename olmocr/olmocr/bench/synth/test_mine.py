import unittest

from .mine_html_templates import generate_tests_from_html

class TestMineTests(unittest.TestCase):
    def test_absent_nested(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Paradigm for Nuclear Safety</title>
</head>
<body>
    <header>
        <div class="logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
    </header>
    
    <main>
        <h1>New Paradigm for Nuclear Safety</h1>
        
        <div class="attribution">
            <p>Thursday, April 25, 2013</p>
            <p>Japan Nuclear Safety Institute</p>
            <p>Shojiro Matsuura, Chairman</p>
        </div>
    </main>
    
    <footer>
        <div class="footer-line"></div>
        <div class="footer-content">
            <div class="footer-text">
                <p class="tagline">In Pursuit of Improved Nuclear Safety</p>
                <p class="copyright">Copyright © 2012 by Japan Nuclear Safety Institute. All Rights Reserved.</p>
            </div>
            <div class="footer-logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
        </div>
    </footer>
</body>
"""
        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertEqual(len([test for test in tests if test["type"]=="absent"]), 2)

    def test_text_basic(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bone Morphology Description</title>
</head>
<body>
    <main>
        <p>The posterior end exhibits a curved process to articulate with the angular. Aside from the process, the rest of the posterior end has slight curvatures for articulation, but is mostly uniform. Ventral border of the bone is mostly straight, with slight curvature (FIG. 20).</p>
        
        <p><span class="section-heading">Lateral</span>- A spine runs from the anterior-most tip, reduces in height ~3/4 way down toward posterior, and terminates at the center of the posterior notch. A fossa is present on the dorsal side of the spine. The posterior end exhibits more relief than in medial view, with the medial side of the posterior process extending past the posterior notch.</p>
        
        <p><span class="section-heading">Ontogeny</span>- Anterior tip is sharply pointed in AR12 and AR1 with AR2 being rounded, though this could be due to breakage. Anterior dorsal margin is straight and flat in AR12; AR2 shows little curvature and AR1 shows the most curvature; curving outward dorsally. Dorsal incisure is anteroposteriorly oriented in AR12, in AR2 there is some ventral curvature, and in AR1 there is a posteroventral curvature. Both AR1 and AR3 are curved on the ventral margin while AR12 is mostly straight. Posterior end of AR1 exhibits four undulations, ventral process is not yet extended. A fossa is present dorsal to the ventral process, not seen on AR12 or AR2. In medial view the lateral ridge is visible posteriorly in AR1 and AR2l the ridge does not fully extend anteriorly. In lateral view of the posterior the ventral process is present on AR2, but not fully extended posteriorly. Tip of the anterodorsal process is sharply pointed in AR1 and AR2, rounded in AR12. A short ridge is present on the dorsal edge of the dorsal process of AR1. The short ridge on the posterodorsal process of AR2 is slightly more ventral than in AR1. On AR12 the ridge is long and positioned most ventral. The lateral ridge is closest to the ventral margin in AR1. In AR2 the ridge is positioned more dorsally and in AR12 the ridge terminates and the anterior tip. The section of bone ventral to the lateral ridge appears to thin with age. The posterior notch on AR12 is curved anteriorly and the medial side of the notch extends posteriorly</p>
    </main>
    
    <footer>
        <p>46</p>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        for test in tests:
            print(test)
