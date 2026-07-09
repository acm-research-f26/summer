"""
Sample DOM element lists for offline ARP evaluation.

These are simplified snapshots from pages used in benchmark/tasks.json.
They let us test the scorer and compare click efficiency without a live browser.
"""

UTD_HOMEPAGE_ELEMENTS = [
    {"index": 0, "tag": "a", "text": "Home", "attrs": {}},
    {"index": 1, "tag": "a", "text": "About", "attrs": {}},
    {"index": 2, "tag": "a", "text": "Academics", "attrs": {}},
    {"index": 3, "tag": "a", "text": "Admissions", "attrs": {"aria-label": "Undergraduate Admissions"}},
    {"index": 4, "tag": "a", "text": "Apply Now", "attrs": {"title": "Start your application"}},
    {"index": 5, "tag": "a", "text": "Research", "attrs": {}},
    {"index": 6, "tag": "a", "text": "Campus Life", "attrs": {}},
    {"index": 7, "tag": "button", "text": "Search", "attrs": {"aria-label": "Open search"}},
    {"index": 8, "tag": "a", "text": "Athletics", "attrs": {}},
    {"index": 9, "tag": "a", "text": "Give", "attrs": {}},
]

WIKIPEDIA_ELEMENTS = [
    {"index": 0, "tag": "input", "text": "", "attrs": {"placeholder": "Search Wikipedia", "name": "search"}},
    {"index": 1, "tag": "a", "text": "Main Page", "attrs": {}},
    {"index": 2, "tag": "a", "text": "Random article", "attrs": {}},
    {"index": 3, "tag": "a", "text": "Python (programming language)", "attrs": {}},
    {"index": 4, "tag": "a", "text": "Today's featured article", "attrs": {}},
    {"index": 5, "tag": "a", "text": "Current events", "attrs": {}},
    {"index": 6, "tag": "button", "text": "Search", "attrs": {}},
]

HN_ELEMENTS = [
    {"index": 0, "tag": "a", "text": "Hacker News", "attrs": {}},
    {"index": 1, "tag": "a", "text": "new | past | comments | ask | show | jobs | submit", "attrs": {}},
    {"index": 2, "tag": "a", "text": "login", "attrs": {}},
    {"index": 3, "tag": "a", "text": "AI startup raises $50M Series B", "attrs": {}},
    {"index": 4, "tag": "a", "text": "Show HN: I built a browser agent", "attrs": {}},
    {"index": 5, "tag": "a", "text": "Why Rust is taking over systems programming", "attrs": {}},
]

# Simulated agent click sequences: baseline tends to explore; ARP picks top-ranked first.
# success = goal_element appears in the click sequence (task completed).
SIMULATED_TRAJECTORIES = {
    "utd_admissions": {
        "task": "Open the UTD website and find when admissions open for fall semester",
        "elements": UTD_HOMEPAGE_ELEMENTS,
        "baseline_clicks": [0, 2, 1, 3, 4],  # wanders through nav
        "arp_clicks": [3, 4],  # goes straight to admissions → apply
        "goal_element": 4,
    },
    "wiki_python": {
        "task": "Go to Wikipedia and find when Python programming language was first released",
        "elements": WIKIPEDIA_ELEMENTS,
        "baseline_clicks": [1, 4, 0, 3],
        "arp_clicks": [0, 3],  # search then python article
        "goal_element": 3,
    },
    "hn_top_story": {
        "task": "Open Hacker News and read the title of the top story on the front page",
        "elements": HN_ELEMENTS,
        "baseline_clicks": [0, 1, 2, 3],
        "arp_clicks": [3],
        "goal_element": 3,
    },
}


def task_succeeded(click_sequence: list[int], goal_element: int) -> bool:
    return goal_element in click_sequence
