"""Check notebook startup dependencies without loading the corpus."""

import numpy as np
from sklearn_extra.cluster import KMedoids  # Used by pam_multilingual.ipynb
from umap.umap_ import UMAP  # Used by plot_comparison_figures.ipynb


def main():
    # The points at 0, 1, 10 and 11 form two well-separated groups.
    points = np.array([0.0, 1.0, 10.0, 11.0])
    distances = np.abs(points[:, None] - points[None, :])
    model = KMedoids(
        n_clusters=2, metric="precomputed", method="pam", random_state=42
    ).fit(distances)

    # Cluster IDs are arbitrary; check the partition instead.
    a, b, c, d = model.labels_
    if not (a == b and c == d and a != c and np.isclose(model.inertia_, 2.0)):
        raise RuntimeError(
            f"Unexpected PAM result: labels={model.labels_}, inertia={model.inertia_}"
        )
    print(f"Startup smoke check passed: KMedoids PAM fit and {UMAP.__name__} import.")


if __name__ == "__main__":
    main()
