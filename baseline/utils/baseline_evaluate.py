import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay


# INPUT: numpy arrays similarities and labels
def roc_det_plot (labels, similarities):
    # Plotting (source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html)
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    # Plot ROC curve
    roc_display = RocCurveDisplay.from_predictions(labels, similarities, ax=ax_roc, name="FaceNet")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) curve")
    ax_roc.grid(linestyle="--")

    # Plot DET curve
    det_display = DetCurveDisplay.from_predictions(labels, similarities, ax=ax_det, name='FaceNet')
    ax_det.set_title("Detection Error Tradeoff (DET)")
    ax_det.grid(linestyle="--")

    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_det_curves.png")