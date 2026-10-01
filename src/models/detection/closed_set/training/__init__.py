"""Training the closed-set RT-DETR detector on your own dataset: ``DetectorTraining`` (``api.py``) reads a COCO or YOLO
folder, trains the way RT-DETR's authors and the large toolkits do, keeps the best epoch, and reports it.

Importing this package costs no torch: ``api``, ``datasets``, ``metrics``, ``plan``, ``recipes`` and ``report`` are
torch-free, and ``trainer`` is imported only when a run starts.
"""
