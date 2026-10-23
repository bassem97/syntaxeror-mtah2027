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

### External training data

The classifier of §5 is trained on data from other matches only. Two public sources were
converted into our feature layout with a per-frame H/A/D label; the submission uses the first.

**SkillCorner open data** (https://github.com/SkillCorner/opendata, MIT, "please credit
SkillCorner"): broadcast tracking of twenty 2024/25 A-League matches at 10 fps, every player
present (extrapolated when out of shot), the ball flagged as detected or not, and a per-frame
possession field naming the team in possession. `possession/skillcorner.py` resamples each
match to 25 fps by linear interpolation inside each period and takes H or A from the possession
field; a frame without a possession is D when at least six players are seen in the broadcast
image (the ball is out of play) and X, not graded, otherwise (a replay or a close-up: about a
third of the frames). Only a ball flagged as detected counts as seen, which leaves 60% of frames
with a ball, close to the challenge tracking's 57%. Our hand-written model agrees with these
labels at spellF1 0.36 and macroF1 0.76 (team right on 82% of alive frames), so they follow the
same convention; the lower spellF1 than on Metrica comes from the ungraded holes cutting
possessions short.

**Metrica Sports' sample games** (https://github.com/metrica-sports/sample-data: tracking at
25 fps and events, anonymised, "please acknowledge the source"): `possession/external.py`
derives the label from the events (a pass, carry, recovery, shot, set piece or ball lost puts
the ball with its team; the ball going out, a goal, a foul, a card, an offside or the end of a
half makes it dead until the next set piece, goal kicks included). The hand-written model agrees
with those labels at spellF1 0.50 and 0.52 on games 1 and 2, the same range as on the challenge
match; game 3's events record few fouls and its labels under-count stoppages, so it was left out.
Trained on games 1 and 2 the classifier was our first submitted version (§5).

**The DFL / Sportec open dataset** (Bassek et al. 2025, CC BY 4.0; `possession/sportec.py`):
seven Bundesliga matches of optical full-pitch tracking at 25 fps with the provider's own
per-frame ball status and possession team, the closest tracking to the challenge's of any public
source. It does not help, and the reason is instructive: the DFL convention marks the ball dead
on 41 to 52% of frames and changes possession on almost every touch, and our hand-written model
agrees with it at spellF1 0.25 to 0.31 (macroF1 0.74 to 0.81), against 0.36 with SkillCorner's
possession field and 0.55 with the organisers' truth. A classifier trained on it alone scores
0.573 on the ten validation windows, below the Metrica one; added to the SkillCorner matches it
lowers 0.625 to 0.604. Across the four sources tried, the gain on the challenge match follows the
agreement of a source's labels with the organisers' convention, not the fidelity of its tracking.

What did not help on the SkillCorner set either: larger classifiers (600 trees of 31 leaves,
1000 of 63: the public window falls to 0.40-0.51), five seeds instead of three (identical to the
digit), the Metrica classifier as a second evidence term, and refitting any hand-written weight
next to the learned evidence (all within noise; less carry or less hand evidence overall lose).

### SoccerTrack v2: tried, no gain

We also converted six of the ten matches of SoccerTrack v2 (CC BY 4.0, full-pitch player
positions at 25 fps with team-labelled ball actions; `possession/soccertrack.py`). Its released
pitch positions sit on a 1.05 m grid and its ball is reconstructed between actions, so we
rebuilt the ball from the tracks (at the actor at each action, at the carrier's feet between two
actions by the same player, unseen while out of play) and blanked it in runs drawn from the
challenge tracking's gap lengths. Even so our hand-written model agrees with its labels at
spellF1 0.30 and macroF1 0.65, against 0.50 and 0.85 on Metrica, and a classifier trained on it
adds nothing: alone it scores below the Metrica one on every measure, and added to the Metrica
games it moves the ten validation windows from 0.576 to 0.579 spellF1 while losing 0.01 macroF1.
The data is not used.

### A second ball track from the video: built, then withdrawn

The tracker loses the ball on 41% of frames, so we trained a small heatmap detector on the
tactical video to fill those frames, using the provided tracking projected through the
homography as its labels. It worked (held-out localisation within 8 px on 82% of frames) and
helped a little, but the organisers confirmed that labelling the provided match to train a
model is not allowed, whatever the labels are derived from, so it is not part of the
submission and its code is not in this repository. A detector trained on external, documented
data would be allowed; we did not build one.

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

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| SyntaxEror (hand evidence + classifier trained on SkillCorner's matches) | 0.865 | 0.648 | 12 |
| hand evidence only (`python -m possession.model`) | 0.911 | 0.606 | 18 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

On the full match the file contains 213 possessions of mean 18 s and 99 stoppages of mean 26 s,
with 41% of graded frames dead (hand evidence only: 206, 99, 41%). Of the 17 true possessions, the two shorter than 1.5 s are absorbed
by their neighbours; the other 15 are found, and where one falls below the 0.7 IoU bar it is a
boundary one to three seconds off, not a missing possession.

## 5. Fitting the decoder on our own labels

The public window cannot rank settings: 17 possessions in the fastest five minutes of the match,
and the organisers say as much. We hand-labelled nine more windows with the kit in
`labeling/` (first half minutes 13, 26 and 38, second half minutes 4, 10, 20, 37 and 44, plus the
3.5 minutes after the public window), touch-level, following the convention written in `labeling/README.md`. The
labeller first labelled the public window blind and scored 0.96 macroF1 against the truth. With
the public window that is 48.5 minutes and 107 possessions. The labels are validation data only;
they are never an input to the algorithm.

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

On those seven windows the hand-evidence decode scores spellF1 0.592 (5-95% bootstrap 0.51 to
0.68) and macroF1 0.855; the detector variant 0.544 (0.46 to 0.63) and 0.856. The submitted
file adds the learned evidence below.

### The same fit on the external games

`analysis/fit_external.py` runs the same grid against the event-derived labels of the three
Metrica sample games (190 minutes), which uses nothing from the challenge match. It confirms
the two changes that matter most: the tempo floor (mean spellF1 over the games 0.493 without,
0.518 with) and the 1.2 m control radius (best at every other setting). It is flat on the
dead-ball cost and the switch delay, the latter expected since event times are only good to a
second. It disagrees on one value: it prefers a team-switch cost of 30 to 45. That preference is
a property of those labels, whose possessions are twice as fragmented as the organisers' truth
(median 5 s against 11 s, because every ball lost and recovery is an event), and the setting it
picks scores 0.471 on our ten windows against 0.546 for the current one.

Should the organisers rule that parameter values may not be chosen against our own labels of
this match, the fallback is the setting fitted on the external games alone (team cost 30, dead
cost 40, control radius 1.2 m, carry 0.25, tempo floor, no delay), which with the classifier
scores spellF1 0.524 and macroF1 0.809 on the ten windows and 0.562 / 0.870 on the public
window; the values set by hand before any labelling score 0.500 / 0.795 and 0.521 / 0.850.

### A learned evidence term, trained on other matches

A classifier trained on our hand labels of this match lifted held-out spellF1 from 0.546 to
0.596, and the organisers' clarification ruled it out: a model may not be trained on annotations
of the provided match. The same recipe trained on other matches (§2) is allowed, and is what the
submission uses.

`possession/learned.py` fits a gradient-boosted classifier (LightGBM, 300 trees of 15 leaves,
strong regularisation, three seeds averaged) that gives every frame a probability of H, A and D
from 90 columns: the tracking features, the hand-written evidence terms, and centred rolling
means of each over 1, 2, 4 and 10 seconds. Frames within 25 of a label transition are left out
(label times are coarse), every second frame is kept (neighbours are near-duplicates), and every
match is also fed mirrored (home and away swapped) so no team bias can be learned. Its
log-probabilities, at weight 0.35, are added to the hand-written evidence and decoded by the
same Viterbi with the same costs and switch delay. Nothing from the challenge match enters
training, so every number below is a fair test.

| evidence | public window spellF1 / macroF1 | ten hand-labelled windows, pooled spellF1 (5-95%) / macroF1 |
|---|---|---|
| hand-written | 0.606 / 0.911 | 0.546 (0.48-0.62) / 0.805 |
| hand + classifier trained on Metrica games 1-2, x0.15 | 0.600 / 0.882 | 0.576 (0.52-0.63) / 0.825 |
| hand + classifier trained on SkillCorner's 20 matches, x0.35 (submitted) | 0.648 / 0.865 | 0.625 (0.56-0.69) / 0.837 |
| hand + classifier trained on both, x0.40 | 0.634 / 0.864 | 0.609 (0.55-0.67) / 0.844 |
| SkillCorner classifier alone, x0.5 | 0.317 / 0.740 | 0.563 (0.47-0.64) / 0.802 |

The SkillCorner classifier's gain holds on a plateau of weights from 0.30 to 0.45 (0.608 to
0.625 pooled) and is the first external source whose gain exceeds the one the in-match
classifier gave. Eight of the ten windows improve; the one that falls is the window holding the
penalty, where the broadcast-trained classifier reads the two-minute stoppage as play. Adding
the Metrica games to the training set buys 0.007 of macroF1 and costs 0.016 of spellF1, the
ranking metric, so they are left out. The classifier alone under-segments the fast public window
exactly as the in-match one did; the hand control cue is kept for that reason.

The three trained models are in `models/`, decoding from them is deterministic, and
`python -m possession.learned --external <folder> --fit models/learned` retrains them from the
converted matches.

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

Reads the three tactical files named in the README, from `./data` or `$MTAH_DATA_DIR`, and the
three classifier files in `models/`. The run is deterministic; a clean clone on an empty data
folder produces a byte-identical file. `python -m possession.model` writes the hand-evidence-only
decode. Retraining the classifier needs SkillCorner's open data (the twenty `*_tracking_extrapolated.jsonl`
and `*_match.json` files, one folder per match): `python -m possession.skillcorner --root
<folder> --out ../external-cache`, then `python -m possession.learned --external
../external-cache --fit models/learned`.
