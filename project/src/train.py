"""
quick summary of everything else we tried first

1. random forest / LightGBM (tree ensembles): score 81%  at best. learnt that trees cant
  extrapolate, they can only predict values inside the range they saw during training

2. a small neural net: tried a couple hidden layer configs. also too many hyperparameters 
   to tune. + dataset this small, not worth it.

3. logistic regression: scored BETTER than LDA on the batches its actually trained on, 
   but does not generalize well. LDA assumes one shared covariance matrix across all classes.

4. LDA + LR averaged together: thought it might work and it sounds reasonable because they fill
   the gaps between the two but it made the score worse than LDA alone. 
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer  # tried QuantileTransformer didnt perform any better
import data
import validate
from features import features

PROXY_BATCHES = (9, 7)

# giving the model a chance to correct itself
ROUNDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def rounds_schedule(n: int | None) -> tuple[float, ...]:
    """ROUNDS as written, unless --rounds asks for a different count.

    with a count, the endpoints of ROUNDS are kept and n values are spread evenly
    between them, handy for a quick sweep without editing the tuple by hand.
    """
    if n is None:
        return ROUNDS
    if n == 1:
        return (sum(ROUNDS) / len(ROUNDS),)
    return tuple(np.linspace(ROUNDS[0], ROUNDS[-1], n))


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.001) # shrinkage=0.001 was manually  tuned, tried many different combinations to find the plateau


def lr():
    # not good
    return LogisticRegression(C=0.5, max_iter=5000)


def prepare(df_source, df_target):
    Xs, fill = features(df_source)
    Xt, _ = features(df_target, fill)
    scaler = StandardScaler().fit(Xs)
    return scaler.transform(Xs), scaler.transform(Xt), df_source[data.TARGET_COL].to_numpy()


def self_train(make, Xs, ys, Xt, rounds):
    """Retrain on the target rows the model is most confident about"""
    clf = make().fit(Xs, ys)
    P = clf.predict_proba(Xt)
    classes = clf.classes_

    for frac in rounds:
        pred, conf = classes[P.argmax(1)], P.max(1)

        keep = np.zeros(len(Xt), bool)
        for c in classes:
            idx = np.flatnonzero(pred == c)
            if len(idx):
                keep[idx[np.argsort(-conf[idx])[: max(1, int(frac * len(idx)))]]] = True

        clf = make().fit(np.vstack([Xs, Xt[keep]]), np.concatenate([ys, pred[keep]]))
        P = clf.predict_proba(Xt)

    return P, classes


def run(df_source, df_target, method="selftrain", rounds=()):
    """Return (probabilities, classes) for the target frame."""
    Xs, Xt, ys = prepare(df_source, df_target)

    if method == "baseline":
        clf = lr().fit(Xs, ys)
        return clf.predict_proba(Xt), clf.classes_

    Pl, classes = self_train(lda, Xs, ys, Xt, rounds)
    # Pr, _ = self_train(lr, Xs, ys, Xt, rounds)
    # return (Pl + Pr) / 2, classes
    return Pl, classes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="selftrain", choices=["selftrain", "baseline"])
    p.add_argument("--rounds", type=int, default=None, help="override the length of ROUNDS for a quick sweep")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rounds = rounds_schedule(args.rounds)
    train = data.load_train()
    
    for k in PROXY_BATCHES:
        source = train[train[data.BATCH_COL] != k]
        held = train[train[data.BATCH_COL] == k]
        y = held[data.TARGET_COL].to_numpy()

        P, classes = run(source, held, args.method, rounds)
        score = validate.macro_f1(y, classes[P.argmax(1)])
        print(f"held-out batch {k}: macro-F1 {score:.4f}  ({len(held)} rows)")

    if args.no_submit:
        return 0

    test = data.load_test()
    P, classes = run(train, test, args.method, rounds)
    pred = classes[P.argmax(1)]

    path = data.write_submission(test[data.ID_COL], pred, args.out)
    counts = np.bincount(pred, minlength=7)[1:]
    print(f"\nwrote {path}")
    print("predicted counts:", counts.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
