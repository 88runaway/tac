| 配置文件                        | 视觉相机                     | 触觉   | 视觉 backbone 学习率 | 触觉 backbone |
| --------------------------- | ------------------------ | ---- | --------------- | ----------- |
| `train_config`              | `cam_high`               | ✅ 左右 | `1e-5`          | `1e-5`      |
| `train_config_vision`       | `cam_high`               | ❌    | `1e-5`          | `0`         |
| `train_config_vision_all`   | `cam_high` + `cam_wrist` | ❌    | `1e-5`          | `0`         |
| `train_config_all`          | `cam_high` + `cam_wrist` | ✅ 左右 | `1e-5`          | `1e-5`      |
| `train_config_freeze`       | `cam_high`               | ✅ 左右 | `1e-5`          | `0（冻结）`     |
| `train_config_tactile_full` | 无                        | ✅ 左右 | `0`             | `1e-5`      |
| `train_config_scrach`       | `cam_high`               | ✅ 左右 | `1e-5`          | `1e-5`      |
