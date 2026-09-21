Changelog
---------

0.1.0 (2026-04-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of LeggedManip Lab
* Support for 7 legged robot platforms with manipulator arms (GO2-ARX5, GO2-PIPER, GO1-ARX5, GO1-WX250S, B2-Z1, B1-Z1, AGO-Z1)
* Two training modes per platform: Flat and WBC (Whole-Body Control)
* RSL-RL PPO training pipeline with curriculum learning and domain randomization
* Standardized observation and action spaces across all platforms
* MuJoCo deployment scripts for trained policy inference
