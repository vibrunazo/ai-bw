# Gary

Gary is a StarCraft: Brood War AI built to be a **sparring partner**. It plays like a real
opponent: pro-level game sense, but human hands, human limits and human mistakes. You can tell
it what to play and practice against it as many times as you like. And you can ask it why it
did what it did.

**License:** code is [MIT-0](LICENSE); docs and data are [CC0](LICENSE-CC0). No attribution required.

Gary is meant to play **every race in every matchup**. TvZ came first because it was convenient
to start somewhere, not because Gary is a Terran bot; the learned Zerg that plays it in self-play
is Gary too, playing Zerg.

**Status:** early. So far Gary plays Terran and Zerg in TvZ, with build and army decisions learned
from pro replays, but it's not a good sparring partner yet. What each version can do:
[docs/PROGRESS.md](docs/PROGRESS.md). The design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
