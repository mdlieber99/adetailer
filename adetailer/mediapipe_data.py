"""Face mesh landmark connections, vendored from mediapipe's legacy solutions.

Copied verbatim from ``mediapipe.python.solutions.face_mesh_connections`` (the
legacy ``mp.solutions`` API, removed in mediapipe 0.10.30+).  The Tasks API uses
the same 468/478 point face mesh topology, so the indices transfer directly;
they are kept here as literal data so the Tasks code path never depends on a
mediapipe attribute that may be moved or deleted again.

Verified identical to ``mediapipe.tasks.python.vision.FaceLandmarksConnections``
(``FACE_LANDMARKS_FACE_OVAL`` / ``LEFT_EYE`` / ``RIGHT_EYE``) on mediapipe 1.0.1
and to ``mp.solutions.face_mesh`` on mediapipe 0.10.21.
"""

from __future__ import annotations

FACEMESH_FACE_OVAL: frozenset[tuple[int, int]] = frozenset(
    [
        (10, 338),
        (21, 54),
        (54, 103),
        (58, 132),
        (67, 109),
        (93, 234),
        (103, 67),
        (109, 10),
        (127, 162),
        (132, 93),
        (136, 172),
        (148, 176),
        (149, 150),
        (150, 136),
        (152, 148),
        (162, 21),
        (172, 58),
        (176, 149),
        (234, 127),
        (251, 389),
        (284, 251),
        (288, 397),
        (297, 332),
        (323, 361),
        (332, 284),
        (338, 297),
        (356, 454),
        (361, 288),
        (365, 379),
        (377, 152),
        (378, 400),
        (379, 378),
        (389, 356),
        (397, 365),
        (400, 377),
        (454, 323),
    ]
)

FACEMESH_LEFT_EYE: frozenset[tuple[int, int]] = frozenset(
    [
        (249, 390),
        (263, 249),
        (263, 466),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        (384, 398),
        (385, 384),
        (386, 385),
        (387, 386),
        (388, 387),
        (390, 373),
        (398, 362),
        (466, 388),
    ]
)

FACEMESH_RIGHT_EYE: frozenset[tuple[int, int]] = frozenset(
    [
        (7, 163),
        (33, 7),
        (33, 246),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
        (157, 173),
        (158, 157),
        (159, 158),
        (160, 159),
        (161, 160),
        (163, 144),
        (173, 133),
        (246, 161),
    ]
)
