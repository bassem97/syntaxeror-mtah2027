# SyntaxEror — Ball possession from automatic tracking

More than a Hack 2027, Phase 1. Team SyntaxEror. *Draft.*

## 1. Approach

We treat possession as a sequence of segments rather than a sequence of frames. Each video frame
gets a score for each of the three states (home `H`, away `A`, dead ball `D`) from a handful of
interpretable evidence terms computed from the tracking and the Smart Tagging. A Viterbi decoder
then finds the labelling of each half that maximises the total score minus a fixed cost for every
state change. The switching cost is the whole trick: per-frame evidence flickers, and the cost
means a change of possession has to be backed by sustained evidence before it is worth paying for,
so the output is made of whole possessions.

We chose this over a per-frame classifier plus smoothing because the ranking metric, `spellF1`,
scores possessions as objects. The organisers' own baseline shows the problem: nearest player to
the ball plus a majority vote reaches a reasonable per-frame score while producing 669 possessions
for a match that has a few hundred. A decoder with an explicit switching cost reasons about segment
boundaries directly, and every term in it is something a coach would recognise.

No learned model is involved. There are about twenty parameters, all command-line flags, and the
public 5-minute ground truth was used to check that each evidence term does what we think, not to
fit the parameters (section 5).

### What the ground truth turned out to be

The public window shows that the corrected tracking changes the owning team at a **touch**: there
are possessions of 19 and 33 frames, and most switches sit at a kick or a reception by the new
team, sometimes while the tracked ball is still several metres from the player, which is what an
aerial ball looks like once projected onto the pitch. This shaped the team evidence: proximity
alone switches a second too early, so touches are a separate, sharp term.

### Evidence terms

**Alive versus dead** is decided by dead-ball evidence alone. The two team scores are built so that
the better of them is always zero, so team evidence never argues with a stoppage. This mattered: a
goalkeeper standing on the ball before a goal kick gives strong home evidence, and in an earlier
version that outvoted the stoppage.

Dead-ball evidence, positive means dead:

| term | signal | why |
|---|---|---|
| out | ball tracked outside the pitch, then held for 3 s | the one unambiguous cue, and it marks the stoppage instant; nobody restarts within 3 s |
| speed | mean player speed over 1 s, relative to a rolling 3-minute median floored at the match median | play stops, players slow down. Relative, because the match's tempo drifts a lot: the public window is the fastest five minutes of the match. Floored, because a stoppage longer than the window drags the median down to walking pace and then looks like play |
| far | the further team's nearest player is more than 13 m from the ball | set-piece geometry, capped low because a low block during build-up looks the same |
| placed | ball still (under 1.5 m/s) with nobody within 1.5 m | ball placed for a restart |
| goal | from the goal moment to the next kick-off tag | the `GOALS` tag starts during the build-up, so the window starts at the ball seen over the line or, failing that, its last fast movement before it disappears into the net |
| moving | ball travelling faster than 4 m/s | a fast ball is in play; a dead ball is still, carried or lost |
| kick | ball goes from under 2.5 m/s to over 8 m/s within a second | a kick from standstill is a restart or a pass, either way play; strong alive evidence for the next 2 s so restarts are decoded at the kick |
| invisible | ball not tracked | mild dead evidence |

Team evidence, positive means home:

| term | signal |
|---|---|
| control | the gap between the nearest away and nearest home player, only on frames where someone is within 1.2 m of the ball |
| carry | while nobody controls the ball, a weak vote for the last team that did, with no time limit |
| touch | a change of ball velocity above 4 m/s within 3.5 m of a player, or the ball reappearing next to one, is an impulse for that player's team. Weight zero in the submitted setting: the fit in §5 found it adds spurious switches |
| shape | home spread minus away spread, on frames without a ball only: the side in possession spreads out, the defender compacts |
| carrier | which team has a nearby player whose velocity matches the ball's: the carrier moves with the ball, a defender beside it does not. Names the right team on 90% of the frames where it has a clear margin, a third of moving-ball frames. Weight zero in the submitted setting: on 34 possessions it added a little macroF1, on 69 it costs 0.025 spellF1 (§5) |
| restart prior | a tagged set piece that names a team (corner, goal kick, free kick, kick-off after a goal: the conceding side) adds a small vote for that team from 2 s before to 15 s after the tag |

The carry term is the challenge's own possession rule written as evidence: a loose or airborne ball
stays with the last team that had it until the other side gains control. Giving it a time limit was
a mistake we measured: once it expired during a long occlusion the decoder switched team at the
first silent frame, four seconds before the real touch.

### Decoding

Viterbi over the three states, per half. Switching between teams costs 45, entering or leaving a
dead ball costs 40, in the units of the evidence above (typical per-frame evidence is between 0.5
and 2). Dead-ball runs shorter than one second are merged into the following possession: a
sub-second stoppage is a decoding shortcut, not football. Frames before kick-off, at halftime and
after the final whistle are `X`; the graded frames of the second half before its kick-off are `D`,
as the organisers' FAQ states. Finally every team-to-team switch is moved 20 frames later: the
decoder switches when the opponent closes on the ball, the truth at the controlled touch, a
median 0.96 s later on the labelled windows (§5).

## 2. Assets used, and dropped

**Used.** The tactical tracking (players and ball) and the Smart Tagging.

**Dropped.** The homography is not used: we never go back to pixels. Neither video is used. The
broadcast recording and its tracking are not used: it is on its own timeline with no provided
mapping, and the tactical tracking already contains everything our evidence terms need.

**What the Smart Tagging turned out to be worth.** Less than hoped for dead-ball detection. Only 59
restarts are tagged and the tags are coarse: in the public window one free-kick tag starts twelve
seconds before the actual restart, and two of the four stoppages carry no tag at all. The `GOALS`
tags are 70 to 250 seconds long and start during the build-up. We use the tags for two things only:
the goal-to-kick-off window, anchored on the goal moment found in the ball track, and the restart
prior on the team. The prior is worth about one point of `spellF1` on the public window.

**Stoppages after fouls are entered late, and we could not fix it.** Against our labelled windows
the model enters dead ball 1 to 7 s after the whistle on fouls, because at the whistle the ball
is still rolling at 8 to 12 m/s and players are still sprinting, so every cue says play. Three
attempts failed on measurement: a dead prior in the seconds before a tagged free kick or throw-in
(most late stoppages have no tag within reach; no change), moving a predicted stoppage back to the
last touch (helps the two worst cases, hurts five others, since the last touch is 1 to 3 s before
the whistle), and "nobody is chasing the rolling ball" (three times more common in live play,
because passes travel with nobody near them). The whistle is simply not observable in tracking.

**Tried and dropped.** A reception detector (a fast ball slowing onto a player, `touches.receptions`):
it fires about four times a minute and names the right team nine times in ten, but it changes no
label, because a reception is exactly the frame where the control term already fires. The
signature is also present at only three of the twelve team switches in the public window; the
others are kicks, which the touch detector covers, or headers the projected ball cannot show.
Collective deceleration of all players as a whistle cue: only one of the four
public stoppages shows it. Holding the out-of-bounds cue until the next kick: throw-ins have no
kick from standstill, so the hold ran for half a minute. Distance-weighted control instead of a
radius: it only lengthened possessions. Team shape on every frame: it is a slow signal and merged
possessions, so it is restricted to frames without a ball.

### A second ball track from the video (work in progress)

The tracker loses the ball on 41% of graded frames, and those are the frames that matter: duels,
crowded boxes, aerial balls. We train a small ball detector on the tactical video and use it only
where the tracker has nothing. Everything is in `vision/`.

- **Labels for free.** On the 113,000 frames where the tracker has the ball, projecting it through
  the frame's homography gives a pixel position. A numeric check against the image (the ball is
  the strongest small bright blob on grass) shows the projection is unbiased but noisy: on
  isolated balls it is more than 8 px off about 40% of the time. Labels snap to the blob where one
  is clear, otherwise keep the projection.
- **Detector.** A 300k-parameter fully convolutional heatmap network in plain PyTorch, stride 4,
  trained on 256 px crops (14k with a ball, 14k without, plus 20k hard negatives mined from the
  detector's own false peaks: socks, boots, line crossings). Runs on full 1080p frames and emits
  the five strongest peaks per frame.
- **Gap filling.** The tracker knows the ball before and after every gap, so the detector's peaks
  are tracked through the gap by dynamic programming with a motion-smoothness cost, anchored at
  both ends. On hidden tracked stretches it recovers 85% of frames, 82% of them within 32 px, about
  a metre. Long gaps of 15 to 30 s are almost all stoppages; the valuable gaps are the 5 to 10 s
  ones in live play.
- **Label quality was the real problem.** The first detector's localisation plateaued at 59%
  from the first epoch whatever it was trained on. A human check of 46 isolated crops explained
  it: the blob-refined point was on the ball in 28 of the 30 crops where a ball was visible, but
  in 16 crops there was no ball anywhere near the tracked position. Dropping frames without a
  clear blob (29% of them) took held-out localisation to 82%, 86% on isolated balls.
- **Effect, whole match.** The filler's confidence threshold matters: at 0.3 the fills add
  fragmentation and cost spellF1, at 0.4 they help. With the 0.4 fill and the decoder of §5 on
  the internal validation set (33.5 min, 69 possessions): macroF1 0.855 to 0.856, spellF1 0.592 to
  0.544. On the public window spellF1 0.606 to 0.627 and macroF1 0.911 to 0.895. Full-match
  possessions 206 to 216. Before the decoder was fitted the detector led on spellF1 by 0.013 on
  34 possessions; with the fitted decoder it trails. The 5-95% bootstrap intervals on the internal
  set are 0.46 to 0.63 with the detector and 0.51 to 0.68 without: the extra ball positions add
  short spurious possessions that the decoder no longer smooths away.

The detector is therefore a modest lever, not the decisive one. Its ceiling is the frames it
cannot see at all: a ball hidden in a crowd or in the air is lost to the detector exactly as it is
to the tracker. A test set of hand-clicked balls on frames the tracker lost would show how much
is left there.

## 3. Possession rules implemented

- A change of possession needs the other team to gain control. Operationally: a touch by an
  opponent, or enough frames with an opponent within 1.2 m of the ball and closer than our own
  players, to pay the switching cost, which comes to roughly one to two seconds of clear control.
- A loose ball, a deflection or an aerial duel without control does not change possession: while
  nobody is within 1.2 m, the carry term keeps voting for the last controlling team, indefinitely.
- Dead ball is a stoppage, never a possession. Ownership of a dead ball is not predicted.
- Restarts: the kick cue places the end of the stoppage at the kick; the restart prior biases the
  first seconds towards the tagged team.

## 4. Results on the public window

The public window is one of the ten windows the classifier of §5 is trained on, so the
submitted file's score there is in-sample and says nothing. The honest number is the held-out
one: a classifier trained on the other nine windows scores spellF1 0.384 and macroF1 0.871 on
it, against 0.606 and 0.911 for the hand-written evidence alone. This window is the fastest five
minutes of the match and the classifier under-segments it (8 possessions for 17); on the nine
other windows, 90 possessions, it wins. §5 has the full table and the reasoning.

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| SyntaxEror, hand evidence only (`python -m possession.model`) | 0.911 | 0.606 | 18 |
| SyntaxEror, submitted (hand + learned), held out on this window | 0.871 | 0.384 | 8 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

On the full match the submitted file contains 186 possessions of mean 19 s and 99 stoppages of
mean 28 s, with 44% of graded frames dead (the hand-only decode: 206, 99, 41%). Of the 17 true possessions, the two shorter than 1.5 s are absorbed
by their neighbours; the other 15 are found, and where one falls below the 0.7 IoU bar it is a
boundary one to three seconds off, not a missing possession.

## 5. Fitting the decoder on our own labels

The public window cannot rank settings: 17 possessions in the fastest five minutes of the match,
and the organisers say as much. We hand-labelled nine more windows with the kit in
`labeling/` (first half minutes 13, 26 and 38, second half minutes 4, 10, 20, 37 and 44, plus the
3.5 minutes after the public window), touch-level, following the convention written in `labeling/README.md`. The
labeller first labelled the public window blind and scored 0.96 macroF1 against the truth. With
the public window that is 48.5 minutes and 107 possessions. The labels are validation data only;
they are never an input to the algorithm, which the rules allow.

`analysis/fit.py` decodes the model over a grid of its five main parameters (team-switch cost,
dead-switch cost, control radius, carry weight, touch weight) and scores every window with the
organisers' metric code, choosing each window's setting on the other windows only, so the
report is an out-of-sample estimate.

- With three windows the fit generalised worse than the defaults (0.443 against 0.492 held out),
  and nothing was changed.
- With four windows every held-out fit, with both ball sources, moved the same way: control
  radius 1.2 m instead of 1.5, touch impulses off, carry weight 0.15 instead of 0.25, dead-switch
  cost 40 instead of 30. Held-out spellF1 0.553 against 0.499 with the tracker ball. The public
  window is one of the four and weighs a quarter of the fit; choosing on the other three alone
  picks the same setting.
- The last window (second half, minute 37: 9 possessions, 47% dead) was labelled after the fit and
  played no part in it. On it the fitted setting scores spellF1 0.400 against 0.300 for the old
  defaults with the tracker ball, and 0.348 against 0.320 with the detector.
- A finer grid pushed further in the same direction (team cost 80, radius 0.9 m, carry 0.05)
  scores higher in sample and worse held out (0.514), so the coarse pick stands.
- The sixth window (first half, minute 38) holds a penalty: a 2.4-minute stoppage with the ball
  invisible and every player walking, which the model left half alive. The speed cue compares
  speed with a rolling 3-minute median, and after a minute of standing the median itself had
  fallen to walking pace. Flooring the median at the match-wide median fixes it: that window
  goes from spellF1 0.26 to 0.33 and macroF1 0.59 to 0.70, one other window gains, the four
  others and the public window are unchanged to the digit, and the match gains three stoppages.
  Floors above the match median keep buying that window at the cost of the stoppage-heavy
  second-half window, so the floor stays at 1.0.
- Refitting the five decoder parameters on all seven windows leaves the current setting at the
  top of the grid (mean per-window spellF1 0.544 against 0.548 for the best cell, inside the
  noise), and the leave-one-window-out choices score 0.503 held out against 0.544 for the
  current setting. The fit has converged on these labels.
- A second round over the cue weights set by hand before any labels existed (shape, carrier,
  restart prior, speed ratio, moving-ball, kick, out-of-bounds hold, far-team radius, minimum
  dead run), in three grids with the decoder held at its fitted values: every one is flat or
  worse held out, except the carrier cue, whose removal lifts the mean per-window spellF1 from
  0.544 to 0.580 (held out 0.569) and macroF1 from 0.818 to 0.832. It had been accepted on 34
  possessions for a small macroF1 gain; on 69 it adds short wrong-team possessions.
- An error budget over the seven windows (`analysis/error_budget.py`): of 69 true possessions
  42 are found at IoU 0.5; of the 27 missed, 13 are the right team with the boundaries off, 8 are
  shorter than 3 s, 5 are given to the other team, 1 to a dead ball. The boundary offsets have one
  systematic component: a team-to-team switch is predicted a median 0.96 s before the truth,
  because the decoder reacts to the opponent closing on the ball and the truth to the controlled
  touch. Moving every such switch 20 frames later takes pooled spellF1 from 0.580 to 0.592 with
  five windows up, one flat and one down; 30 or 40 frames gain no more. Entries into dead ball
  have no systematic offset, only the known late ones after fouls and some early ones where the
  decoder loses the team before the whistle.

The gain is almost entirely fewer spurious short possessions: on the four fitted windows the
predicted segment count fell from 80 to 70 and the spurious ones from 54 to 43 with the matched
count unchanged; full-match possessions fell from 279 to 225. The direction is the one the public
window alone also rewards, which is why we had refused to follow it before the labels existed.

On the seven windows the submitted file scores spellF1 0.592 (5-95% bootstrap 0.51 to 0.68) and
macroF1 0.855; the detector variant 0.544 (0.46 to 0.63) and 0.856.

### Learned evidence on top of the hand-written terms

With ten labelled windows the labels can train as well as validate. `possession/learned.py`
fits a gradient-boosted classifier (LightGBM, 300 trees of 15 leaves, strong regularisation) that
gives every frame a probability of H, A and D from 90 columns: the tracking features, the
hand-written evidence terms, and centred rolling means of each over 1, 2, 4 and 10 seconds. The
frames within 10 of a hand-labelled transition are left out of training (the labels are only good
to a third of a second there), every window is also fed mirrored (home and away swapped) so no
team bias can be learned, and three seeds are averaged. The classifier's log-probabilities, at
weight 0.35, are added to the hand-written evidence and decoded by the same Viterbi with the same
costs and the same switch delay.

Everything below is held out: each window is scored with a classifier that never saw it, three
seeds averaged. The windows are in the order public, the 3.5 min after it, first-half minutes 26
and 38, second-half minutes 4, 20 and 37, then the last three labelled: first-half minute 13,
second-half minutes 10 and 44.

| evidence | per-window spellF1 | pooled spellF1 (5-95%) | macroF1 |
|---|---|---|---|
| hand-written | 0.606 0.550 0.574 0.500 0.678 0.740 0.463 0.462 0.432 0.487 | 0.546 (0.48-0.61) | 0.805 |
| classifier alone, weight 0.5 | 0.365 0.492 0.511 0.400 0.838 0.880 0.657 0.477 0.517 0.504 | 0.554 (0.47-0.64) | 0.837 |
| hand + classifier x0.35 (submitted) | 0.384 0.720 0.678 0.436 0.781 1.000 0.600 0.587 0.462 0.487 | 0.596 (0.52-0.66) | 0.844 |

The last three windows were labelled after the first version of the classifier was built, so
they were also a fresh test of the committed pipeline before retraining: on them the hand
evidence scores spellF1 0.459 and macroF1 0.692, the pipeline trained on the seven earlier
windows 0.525 and 0.759.

What the classifier adds is where the ball is lost and around stoppages: its most important
inputs are the 10-second means of the team and moving-ball evidence. What it lacks is the hand
control cue's sharpness at quick exchanges, and the public window, the fastest five minutes of
the match, is where that shows: held out, the sum scores 0.384 there against 0.606 for the hand
evidence alone, holding each team about six seconds too long at exchanges. Giving the classifier
only the dead-ball channel keeps that window (0.627) but loses almost all of the pooled gain
(0.550); the team channel is where the gain is. We go with the pooled estimate over 107
possessions and say so here. Dropping the long context windows loses the gain, the weight and the
decoder costs sit on a flat plateau (weights 0.35 to 0.5, costs 40 to 50), single seeds score
within 0.02 of each other, and the labeller's known offset does not matter: shifting the hand
labels by it before training changes nothing.

The hand labels themselves are not distributed with the code; the three trained models are, in
`models/`, and decoding from them is deterministic. They were trained on our own labels of 48.5
minutes of this match, which the rules allow.

## 6. Limitations

- **Team-to-team transition timing.** Restarts are decoded within a second or two, but
  transitions between teams in open play land one to three seconds off in both directions. Proximity
  fires before the touch; touch detection misses soft receptions and misattributes about one touch
  in five; headers happen while the projected ball is metres from everyone.
- **Short possessions.** Possessions under about three seconds are not recoverable with this
  switching cost and are absorbed by their neighbours. The ground truth has several per five
  minutes.
- **Stoppages with the ball in play-like motion.** A foul where the ball keeps rolling, or a keeper
  carrying the ball at pace, is the main source of late entries into `D`.
- **The 41% of frames without a ball** are handled through the carry term, the team-shape term and
  the decoder's inertia. There is no ball-position inference.
- **No learning.** The parameters are hand-set. With a larger labelled sample the evidence weights
  and costs could be fitted properly.

## 7. Reproduction

```bash
pip install -r requirements.txt     # numpy, pandas, lightgbm
python make_submission.py           # -> out/SyntaxEror.csv
```

Reads only the three tactical files named in the README, from `./data` or `$MTAH_DATA_DIR`, and the
three classifier files in `models/`. The run is deterministic; a clean clone on an empty data
folder produces a byte-identical file. `python -m possession.model` writes the hand-evidence-only
decode, and `python -m possession.learned --fit` retrains the classifier from the labels.
