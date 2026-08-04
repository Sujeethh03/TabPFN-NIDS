"""Week-1 smoke test — verifies TabPFN installation works."""
import time
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    sys.exit("TabPFN not installed. Run: pip install -r requirements.txt")

print("Loading data...")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Running TabPFN...")
t0 = time.time()
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
elapsed = time.time() - t0

acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")
print(f"Runtime: {elapsed:.2f}s")

if acc > 0.95:
    print("\nSmoke test PASSED. Environment is ready.")
else:
    print("\nSmoke test WARNING: accuracy below 0.95.")
