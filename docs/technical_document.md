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

No learned model is involved. There are about a dozen parameters, all command-line flags, and the
public 5-minute ground truth was used to check that each evidence term does what we think, not to
fit the parameters (section 5).

### Evidence terms

**Alive versus dead** is decided by dead-ball evidence alone. The two team scores are built so that
the better of them is always zero, so team evidence never argues with a stoppage. This mattered: a
goalkeeper standing on the ball before a goal kick gives strong home evidence, and in an earlier
version that outvoted the stoppage.

Dead-ball evidence, positive means dead:

| term | signal | why |
|---|---|---|
| out | ball tracked outside the pitch | the one unambiguous cue |
| speed | mean player speed over 1 s, relative to a rolling 3-minute median | play stops, players slow down. Relative, because the match's tempo drifts a lot: the public window is the fastest five minutes of the match |
| far | the further team's nearest player is more than 13 m from the ball | set-piece geometry, capped low because a low block during build-up looks the same |
| placed | ball still (under 1.5 m/s) with nobody within 1.5 m | ball placed for a restart |
| goal | between a `GOALS` tag and the next kick-off tag | celebration and reset |
| moving | ball travelling faster than 4 m/s | a fast ball is in play; a dead ball is still, carried or lost |
| kick | ball goes from under 2.5 m/s to over 8 m/s within a second | a kick from standstill is a restart or a pass, either way play; strong alive evidence for the next 2 s so restarts are decoded at the kick |
| invisible | ball not tracked | mild dead evidence |

Team evidence, positive means home:

| term | signal |
|---|---|
| control | the gap between the nearest away and nearest home player, only on frames where someone is within 1.5 m of the ball |
| carry | while nobody controls the ball, a weak vote for the last team that did, for up to 5 s |
| restart prior | a tagged set piece that names a team (corner, goal kick, free kick, kick-off after a goal: the conceding side) adds a small vote for that team from 2 s before to 15 s after the tag |

The carry term is the challenge's own possession rule written as evidence: a loose or airborne ball
stays with the last team that had it until the other side gains control.

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
seconds before the actual restart, and two of the four stoppages carry no tag at all. We use the
tags for two things only: the goal-to-kick-off window, and the restart prior on the team. The
prior is worth about one point of `spellF1` on the public window.

## 3. Possession rules implemented

- A change of possession needs sustained control by the other team. Operationally: enough frames
  with an opponent within 1.5 m of the ball and closer than our own players to pay the switching
  cost, which comes to roughly one to two seconds of clear control.
- A loose ball, a deflection or an aerial duel without control does not change possession: while
  nobody is within 1.5 m, the carry term keeps voting for the last controlling team.
- Dead ball is a stoppage, never a possession. Ownership of a dead ball is not predicted.
- Restarts: the kick cue places the end of the stoppage at the kick; the restart prior biases the
  first seconds towards the tagged team.

## 4. Results on the public window

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| SyntaxEror | 0.852 | 0.506 | 17 |
| baseline_smooth125 | 0.581 | 0.424 | 16 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

Per class, dead ball has precision 0.96 and recall 0.80. On the full match the file contains 268
possessions of mean 15 s and 107 stoppages of mean 22 s, with 37% of graded frames dead.

## 5. What we did not tune, and why

Every setting we tried that lengthens possessions (a higher switching cost, a tighter control
radius, a weaker carry) scores higher on the public window and lowers the full-match possession
count, from about 270 to about 225. That is exactly the effect the organisers document with their
own baselines: the window has 17 possessions of mean 14 s and cannot rank anything. We left the
decoder costs and radii where the full-match possession and stoppage counts look like a football
match, and used the window only to check that each evidence term behaves as intended.

## 6. Limitations

- **Team-to-team transition timing.** Restarts are now decoded within a second or two, but
  transitions between teams in open play land one to four seconds off in both directions. The
  nearest-player gap is nearly neutral during quick exchanges, and the ground truth marks the
  change at controlled reception, which we do not detect explicitly. A reception detector (ball
  decelerating onto a player) is the obvious next lever.
- **Short possessions.** Possessions under a second are not recoverable with this switching cost
  and are absorbed by their neighbours.
- **Stoppages with the ball in play-like motion.** A foul where the ball keeps rolling, or a keeper
  carrying the ball at pace, is the main source of missed dead-ball frames.
- **The 41% of frames without a ball** are handled only through the carry term and the decoder's
  inertia. There is no ball-position inference.
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
