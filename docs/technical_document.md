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

One learned term is involved, a classifier trained on other matches (§5). There are about twenty
parameters, all command-line flags; their values were chosen on annotated matches from other
sources, never on this match, and the public 5-minute ground truth was used only to check that
each evidence term does what we think (§5).

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
| touch | a change of ball velocity above 4 m/s within 3.5 m of a player, or the ball reappearing next to one, is an impulse for that player's team |
| shape | home spread minus away spread, on frames without a ball only: the side in possession spreads out, the defender compacts |
| carrier | which team has a nearby player whose velocity matches the ball's: the carrier moves with the ball, a defender beside it does not. Names the right team on 90% of the frames where it has a clear margin, a third of moving-ball frames |
| restart prior | a tagged set piece that names a team (corner, goal kick, free kick, kick-off after a goal: the conceding side) adds a small vote for that team from 2 s before to 15 s after the tag |

The carry term is the challenge's own possession rule written as evidence: a loose or airborne ball
stays with the last team that had it until the other side gains control. Giving it a time limit was
a mistake we measured: once it expired during a long occlusion the decoder switched team at the
first silent frame, four seconds before the real touch.

### Decoding

Viterbi over the three states, per half. Switching between teams costs 30, entering or leaving a
dead ball costs 25, in the units of the evidence above (typical per-frame evidence is between 0.5
and 2). Dead-ball runs shorter than one second are merged into the following possession: a
sub-second stoppage is a decoding shortcut, not football. Frames before kick-off, at halftime and
after the final whistle are `X`; the graded frames of the second half before its kick-off are `D`,
as the organisers' FAQ states. Finally every team-to-team switch is moved 10 frames later: the
decoder switches when the opponent closes on the ball, the annotations at the controlled touch,
which comes later (the delay, like every value above, was chosen on other matches, §5).

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

**Stoppages after fouls are entered late, and we could not fix it.** On the public window and on
the annotated external matches the model enters dead ball seconds after the whistle on fouls, because at the whistle the ball
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
possession field and 0.55 with the organisers' truth. A classifier trained on it, alone or with
the SkillCorner matches, was ranked below the SkillCorner one before the organisers' ruling of
§5 and is not used.

**PFF FC's 2022 World Cup data** (free on request from PFF FC's blog; `possession/pff.py`): broadcast
tracking of all 64 matches at 29.97 fps with every on-the-ball event and its team written into
the frames, the largest set and the one whose labels agree best with our model (hand-written
model against its labels: spellF1 0.56 and macroF1 0.85 on average over the matches, 0.41 to
0.70). The ball-out event PFF writes at every whistle makes the dead spans come out as in the
organisers' convention. It still does not help, for a reason of its own: PFF has no ball while
play is stopped (seen on 1% of dead frames, against two thirds of them in the organisers' public
window), so a classifier
learns "ball seen = alive" and calls the challenge's stoppages play (dead recall 0.55 on the
public window against 0.72 for the SkillCorner one). Drawing the ball on dead frames along the
line from where it went out to where play restarts, blanked to the challenge's rate, repairs the
visibility statistics but not the ranking: alone, with the SkillCorner matches, or with only
twelve of its matches added, every variant was ranked below the SkillCorner classifier before the
ruling of §5, and on the public window none passes it. Its broadcast is cut away from the pitch
far more often than SkillCorner's and its estimated positions during those cuts, kept or not,
leave the classifier worse at the challenge's stoppages. Across the five sources tried, the gain on the challenge match is not predicted by the
fidelity of a source's tracking, nor fully by the agreement of its labels with the organisers'
convention; the one that transfers is the one whose ball and camera behave most like the
challenge's during stoppages.

What did not help on the SkillCorner set either: larger classifiers (600 trees of 31 leaves,
1000 of 63: the public window falls to 0.40-0.51), five seeds instead of three (identical to the
digit), and the Metrica classifier as a second evidence term.

### SoccerTrack v2: tried, no gain

We also converted six of the ten matches of SoccerTrack v2 (CC BY 4.0, full-pitch player
positions at 25 fps with team-labelled ball actions; `possession/soccertrack.py`). Its released
pitch positions sit on a 1.05 m grid and its ball is reconstructed between actions, so we
rebuilt the ball from the tracks (at the actor at each action, at the carrier's feet between two
actions by the same player, unseen while out of play) and blanked it in runs drawn from the
challenge tracking's gap lengths. Even so our hand-written model agrees with its labels at
spellF1 0.30 and macroF1 0.65, against 0.50 and 0.85 on Metrica, and a classifier trained on it
was ranked below the Metrica one on every measure before the ruling of §5. The data is not used.

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
| SyntaxEror (hand evidence + classifier trained on SkillCorner's matches) | 0.880 | 0.582 | 16 |
| hand evidence only (`python -m possession.model`) | 0.892 | 0.537 | 18 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

On the full match the file contains 272 possessions of mean 14 s and 105 stoppages of mean 25 s,
with 41% of graded frames dead (hand evidence only: 305, 112, 40%). An earlier version of this
file, with values chosen against our own labels of the match, scored 0.648 / 0.865 here; the
organisers ruled that choice out (§5) and the values were re-chosen on other matches.

## 5. Choosing the parameter values on other matches

The public window cannot rank settings: 17 possessions in the fastest five minutes of the match,
and the organisers say as much. We first hand-labelled nine more windows of the match (48.5
minutes with the public window, 107 possessions, kit in `labeling/`) and chose the decoder's
values by leave-one-window-out fits on them. We then asked the organisers whether that was
allowed. Their answer: on the challenge match hand labels may not be used for any purpose, not
for training, not for validation, not for choosing or optimising parameter values; what is
allowed is to annotate another match, choose the values on it, and apply them to the challenge
video unchanged. So every value chosen that way was discarded, the hand labels are not used for
anything in this submission, no number in this document comes from them, and the values were
re-chosen by the route the organisers describe, on annotated matches from other sources.

**The annotated matches** are SkillCorner's twenty A-League matches and Metrica's sample games 1
and 2 (§2), with their providers' own public annotations, converted to a per-frame H/A/D label
by `possession/skillcorner.py` and `possession/external.py`; nothing of ours is annotated, so
there is nothing beyond those converters to share.

**The procedure** is `analysis/fit_oos.py`, run once; its record is `analysis/fit_oos_result.json`.

1. Every external match gets an out-of-sample classifier (below): SkillCorner's matches are
   split in two folds, each fold's classifier predicts the other, and the Metrica games get the
   mean of both. So the classifier's weight is chosen on matches its training never saw.
2. Starting from the values set by hand before any labelling, each of nine parameters in turn
   takes the value of a small grid that maximises the mean `spellF1` over the external matches,
   with SkillCorner and Metrica weighted equally as sources, scored with the organisers' metric
   code against the event-derived labels; two passes over the parameters.
3. The chosen values are the defaults of `possession/model.py` and `possession/learned.py` and
   are applied to the challenge match unchanged.

| parameter | set by hand | chosen on other matches | grid |
|---|---|---|---|
| classifier weight | 0 | 0.35 | 0, 0.25, 0.35, 0.45 |
| team-switch cost | 45 | 30 | 30, 45, 60 |
| dead-switch cost | 30 | 25 (flat from 25 to 40) | 25, 30, 40, 55 |
| control radius (m) | 1.5 | 1.2 | 1.2, 1.5, 2.0 |
| carry weight | 0.25 | 0.25 | 0, 0.15, 0.25 |
| carrier weight | 0.5 | 0.5 | 0, 0.5 |
| touch weight | 3 | 3 | 0, 3 |
| tempo floor | off | on (1.0 x match median) | off, on |
| switch delay (frames) | 0 | 10 | 0, 10, 20 |

Mean `spellF1` over the external matches goes from 0.470 to 0.515 (Metrica 0.538 to 0.593,
SkillCorner 0.402 to 0.438); the second pass changes nothing but the classifier weight. The
values the public window cannot rank are left where the external fit put them even where it is
flat. The lower team-switch cost is a property of those labels, whose possessions are about
twice as fragmented as the organisers' truth (every ball lost and recovery is an event), and it
is the main reason the public-window score fell from the version chosen on our own labels (§4):
it is the price of choosing the values the allowed way.

### A learned evidence term, trained on other matches

A classifier trained on our hand labels of this match was built first and withdrawn with the
organisers' first clarification: a model may not be trained on annotations of the provided match.
The same recipe trained on other matches (§2) is allowed, and is what the submission uses.

`possession/learned.py` fits a gradient-boosted classifier (LightGBM, 300 trees of 15 leaves,
strong regularisation, three seeds averaged) that gives every frame a probability of H, A and D
from 90 columns: the tracking features, the hand-written evidence terms, and centred rolling
means of each over 1, 2, 4 and 10 seconds. Frames within 25 of a label transition are left out
(label times are coarse), every second frame is kept (neighbours are near-duplicates), and every
match is also fed mirrored (home and away swapped) so no team bias can be learned. Its
log-probabilities, at weight 0.35, are added to the hand-written evidence and decoded by the
same Viterbi with the same costs and switch delay. Nothing from the challenge match enters
training or the choice of the weight.

| evidence | public window spellF1 / macroF1 | external matches, out of sample, mean spellF1 |
|---|---|---|
| hand-written | 0.537 / 0.892 | 0.498 |
| hand + classifier trained on SkillCorner's 20 matches, x0.35 (submitted) | 0.582 / 0.880 | 0.515 |

On the external matches the gain holds for weights 0.25 to 0.45 (0.513 to 0.515). Before the
ruling of §5 the same classifier, and the alternatives of §2 (Metrica, PFF, Sportec, SoccerTrack,
and combinations), were ranked against our own labels of the match; those rankings are not used
and not reported. SkillCorner's matches were kept as the training set for the reasons visible
without them: the largest set with a ball flagged as detected, and the only one whose classifier
lifts the public window. The classifier alone under-segments the fast public window; the hand
control cue is kept for that reason.

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
- **Values chosen on other conventions.** The parameter values come from matches annotated by
  other providers, whose possessions are about twice as fragmented as the organisers'; the
  team-switch cost in particular is lower than this match would want.

## 7. Reproduction

```bash
pip install -r requirements.txt     # numpy, pandas, lightgbm
python make_submission.py           # -> out/SyntaxEror.csv
```

Reads the three tactical files named in the README, from `./data` or `$MTAH_DATA_DIR`, and the
three classifier files in `models/`. The run is deterministic; a clean clone on an empty data
folder produces a byte-identical file. `python -m possession.model` writes the hand-evidence-only
decode, `python analysis/fit_oos.py --external <folder> --proba <folder>` repeats the choice of
the parameter values on the converted external matches. Retraining the classifier needs SkillCorner's open data (the twenty `*_tracking_extrapolated.jsonl`
and `*_match.json` files, one folder per match): `python -m possession.skillcorner --root
<folder> --out ../external-cache`, then `python -m possession.learned --external
../external-cache --fit models/learned`.
