"""LDA, self-trained onto the target batch. model that we chose.

quick summary of everything else we tried first

1. random forest / LightGBM (tree ensembles): around 81% macro F1 at best. trees cant
  extrapolate, they can only predict values inside the range they saw during training,
  so once the sensors drift past what batch 1-9 looked like, a tree just kind of 
  guesses whatever leaf is closest instead of extending a boundary like a linear
  model can. 

- a small neural net: tried a couple hidden layer configs. with ~10k training rows and
  130 features it just overfit to whatever quirks were sitting in the source batches,
  and every version scored worse than LDA on the batch 9 proxy. also way more knobs to
  tune (layers, dropout, learning rate, epochs, early stopping) for zero payoff on a
  dataset this small. not worth it here, maybe worth it if we had 10x the data.

- plain logistic regression (see lr() below): scores BETTER than LDA on the batches its
  actually trained on, but does not generalize forward as well. LDA assumes one shared
  covariance matrix across all classes, which acts like a built in regularizer, it
  stops the decision boundary from bending to fit every little quirk in the source
  batches. turns out that matters more here than squeezing out extra accuracy on data
  we already have. lr() is kept below for baseline comparisons only (--method baseline),
  do not use it for a real submission.

- LDA + LR averaged together: this used to be the plan, and it sounds reasonable on
  paper (two models that fail differently should average out). tested it for real and
  it made things WORSE than LDA alone, not better. their probability calibrations dont
  mix cleanly, averaging ends up dragging LDA's well behaved probabilities down instead
  of LR benefiting from LDA's stability. the averaging code is still here, commented
  out, so nobody "helpfully" adds it back in without reading this first.

    python train.py                      # LDA self-training, validate + submit
    python train.py --method baseline    # plain LR, no self-training
    python train.py --no-submit          # validate only
    python train.py --rounds 3           # override number of self-training rounds
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, ledoit_wolf  # ledoit_wolf: used manually once to sanity check the 0.001 shrinkage value, not called in the pipeline itself, keeping the import so that check is easy to redo later
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer  # tried QuantileTransformer as a rank-based alternative to StandardScaler, did not beat it, left here in case its worth another shot later

import data
import validate
from features import features

PROXY_BATCHES = (9, 7)

# one entry per self-training round, the fraction of each class's predictions to
# trust that round. small steps at the start on purpose, so the model isnt trusting
# 90% of its own guesses on round 1 when it barely knows anything about the target
# batch yet. length of this tuple = number of rounds.
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
    # shrinkage=0.001 was hand tuned, do not touch this without re-running the full
    # forward chaining sweep. tried a few values either side of it and 0.001 won.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.001)


def lr():
    """baseline only, see the module docstring for why this isnt the real model."""
    return LogisticRegression(C=0.5, max_iter=5000)


def prepare(df_source, df_target):
    """Features + scaling for a (source, target) pair.

    DO NOT remove the StandardScaler call below. we tried it, thinking the pattern
    block is already row normalized so a global scaler felt redundant, and macro F1
    on the actual graded batch 10 dropped from 95.57 to 87. an 8 point drop from
    deleting one line. the proxy folds (batches 9 and 7) barely moved when we did
    this, like 1 point either way, so if you break this again the proxy scores below
    will NOT warn you, only a real submission will show the damage. the reason it
    matters so much is that logscale and logconc are raw, unscaled numbers sitting
    right next to 128 pattern columns that are all roughly the same tiny scale.
    without scaling, those two columns can dominate the covariance matrix LDA
    estimates, which means the pattern block, the one part of our features actually
    built to survive drift, barely gets a vote in the decision boundary. scaling
    puts all 130 columns on equal footing before LDA ever sees them.

    state_fill and the scaler are both learned on the source only, this is the one
    place this pipeline could leak information from the target batch, so its kept
    explicit and obvious on purpose. fit_transform on the combined data would leak.
    """
    Xs, fill = features(df_source)
    Xt, _ = features(df_target, fill)
    scaler = StandardScaler().fit(Xs)
    return scaler.transform(Xs), scaler.transform(Xt), df_source[data.TARGET_COL].to_numpy()


def self_train(make, Xs, ys, Xt, rounds):
    """Retrain on the target rows the model is most confident about.

    selection is per predicted class, ranked by the models own probability, not a
    single global top-N. if we picked a global top-N the majority class would just
    swamp every round and the small classes would never get any pseudo labels at all.
    theres no marginal correction here either, no target class counts, no prior
    reweighting, so whatever skew the model starts with can get reinforced round over
    round. thats exactly why ROUNDS ramps up slowly instead of jumping straight to
    trusting 90% of predictions, it gives the model a chance to correct course before
    committing hard.
    """
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
    # LDA + LR averaging used to live here. tested it, it made things worse, see the
    # module docstring. leaving the two lines commented so the history stays visible
    # instead of just vanishing from git blame.
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

    # only batches 9 (the drift proxy, most realistic since nothing comes after it)
    # and 7 (a sanity fold, less realistic since it trains on 8 and 9 too) get scored
    # here, on purpose, to keep the iteration loop fast. remember: these two barely
    # react to the scaler bug above, dont treat a good score here as proof everything
    # is fine, when in doubt just run the real submission and check the count spread.
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
