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
| speed | mean player speed over 1 s, relative to a rolling 3-minute median | play stops, players slow down. Relative, because the match's tempo drifts a lot: the public window is the fastest five minutes of the match |
| far | the further team's nearest player is more than 13 m from the ball | set-piece geometry, capped low because a low block during build-up looks the same |
| placed | ball still (under 1.5 m/s) with nobody within 1.5 m | ball placed for a restart |
| goal | from the goal moment to the next kick-off tag | the `GOALS` tag starts during the build-up, so the window starts at the ball seen over the line or, failing that, its last fast movement before it disappears into the net |
| moving | ball travelling faster than 4 m/s | a fast ball is in play; a dead ball is still, carried or lost |
| kick | ball goes from under 2.5 m/s to over 8 m/s within a second | a kick from standstill is a restart or a pass, either way play; strong alive evidence for the next 2 s so restarts are decoded at the kick |
| invisible | ball not tracked | mild dead evidence |

Team evidence, positive means home:

| term | signal |
|---|---|
| control | the gap between the nearest away and nearest home player, only on frames where someone is within 1.5 m of the ball |
| carry | while nobody controls the ball, a weak vote for the last team that did, with no time limit |
| touch | a change of ball velocity above 4 m/s within 3.5 m of a player, or the ball reappearing next to one, is an impulse for that player's team |
| shape | home spread minus away spread, on frames without a ball only: the side in possession spreads out, the defender compacts |
| restart prior | a tagged set piece that names a team (corner, goal kick, free kick, kick-off after a goal: the conceding side) adds a small vote for that team from 2 s before to 15 s after the tag |

The carry term is the challenge's own possession rule written as evidence: a loose or airborne ball
stays with the last team that had it until the other side gains control. Giving it a time limit was
a mistake we measured: once it expired during a long occlusion the decoder switched team at the
first silent frame, four seconds before the real touch.

### Decoding

Viterbi over the three states, per half. Switching between teams costs 45, entering or leaving a
dead ball costs 30, in the units of the evidence above (typical per-frame evidence is between 0.5
and 2). Dead-ball runs shorter than one second are merged into the following possession: a
sub-second stoppage is a decoding shortcut, not football. Frames before kick-off, at halftime and
after the final whistle are `X`; the graded frames of the second half before its kick-off are `D`,
as the organisers' FAQ states.

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
- **Effect.** With the filled ball on the internal validation set (13.5 min, 34 possessions):
  macroF1 0.829 to 0.851, spellF1 0.491 to 0.520. On the public window spellF1 0.546 to 0.563 and
  macroF1 0.890 to 0.873. A gain, within the noise of 34 possessions.

What limits it is label quality: the detector's localisation plateaus at the first epoch whatever
the training set, which is the signature of labels that are not on the ball. A human check of the
auto-labels is the next step; a few hundred hand-clicked balls would settle it.

## 3. Possession rules implemented

- A change of possession needs the other team to gain control. Operationally: a touch by an
  opponent, or enough frames with an opponent within 1.5 m of the ball and closer than our own
  players, to pay the switching cost, which comes to roughly one to two seconds of clear control.
- A loose ball, a deflection or an aerial duel without control does not change possession: while
  nobody is within 1.5 m, the carry term keeps voting for the last controlling team, indefinitely.
- Dead ball is a stoppage, never a possession. Ownership of a dead ball is not predicted.
- Restarts: the kick cue places the end of the stoppage at the kick; the restart prior biases the
  first seconds towards the tagged team.

## 4. Results on the public window

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| SyntaxEror | 0.890 | 0.546 | 16 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

On the full match the file contains 279 possessions of mean 14 s and 108 stoppages of mean 22 s,
with 38% of graded frames dead. Of the 17 true possessions, the five shorter than four seconds are
absorbed by their neighbours; the long ones match at IoU 0.75 to 0.95.

## 5. What we did not tune, and why

Every setting we tried that lengthens possessions (a higher switching cost, a tighter control
radius, a weaker carry, a stronger shape term) scores higher on the public window and lowers the
full-match possession count, from about 280 to 220 or fewer. That is exactly the effect the
organisers document with their own baselines: the window has 17 possessions of mean 14 s and
cannot rank anything. We left the decoder costs and radii where the full-match possession and
stoppage counts look like a football match, and used the window only to check that each evidence
term behaves as intended. Moving both costs by 20% in either direction changes the public
`spellF1` by less than the metric's noise band.

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
pip install -r requirements.txt
python make_submission.py           # -> out/SyntaxEror.csv
```

Reads only the three tactical files named in the README, from `./data` or `$MTAH_DATA_DIR`. The
run is deterministic; two runs produce byte-identical output, and the output matches the file we
submitted.
