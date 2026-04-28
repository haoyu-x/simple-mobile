
## Installation


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
cd ~/simple-mobile/simple_mobile
sudo apt install -y cmake build-essential
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo apt-get install ffmpeg
```
```bash
cd ~/simple-mobile/simple_mobile
uv sync
source .venv/bin/activate
```

```bash
cd ~/simple-mobile/simple_mobile/i2rt
uv pip install -e .

cd ~/simple-mobile/simple_mobile/pyroki
uv pip install -e .

```

## Teleop and data collection in MuJoCo:

> [!NOTE]
> 1. Make sure that your iPhone and computer are under the same wifi network.
> 2. Follow [XR Browser useage](https://tidybot2.github.io/docs/usage/#connecting-the-client) for teleop.


Start Teleop to collect data:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python sim_main.py --save
```
Review collected data:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python reviewer.py --sim
```

Sort collected data:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python sort_demos_from_review.py  # data/demos/review_results_20260407_171358.json
```

Next --> [diffusion policy training for mujoco dataset](../diffusion_policy/training_mujoco_dataset.md)

## Teleop and data collection in the real world:

### YAM arms CAN setup

> [!NOTE]
> 1. Remeber to power on the robot arms first.
> 1. Follow [instructions](https://github.com/haoyu-x/simple-mobile/blob/main/simple_mobile/i2rt/docs/getting-started/hardware-setup.md#persistent-can-ids) to set CAN name to can_follower_r and can_follower_l

Quick CAN connection:
```bash
sudo ip link set can_follower_r up type can bitrate 1000000
sudo ip link set can_follower_l up type can bitrate 1000000

```


Test:
```bash
cd ~/simple-mobile/simple_mobile/i2rt
uv run python i2rt/robots/motor_chain_robot.py --channel can_follower_r --gripper_type linear_4310
```

```bash
cd ~/simple-mobile/simple_mobile/i2rt
uv run python i2rt/robots/motor_chain_robot.py --channel can_follower_l --gripper_type linear_4310
```


### Hex mobile base test

> [!NOTE]
> 1. turn on the power
> 1. make sure that the emergency stop switch is released
> 1. make sure your base is connected with the computer with ethernet cable
> 1. make sure to update the HEX_BASE_URL (find on the screen of the base, shown below) in [constants.py](tidybot2/constants.py) to match the Ethernet IP address of your hex mobile base.

<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 16px; margin: 1.5em 0;">
  <img
    src="../docs/assets/home.png"
    alt="right image"
    style="height: 200px; width: auto; object-fit: contain; display: block;"
  />
    <img
    src="../docs/assets/app_selector.png"
    alt="right image"
    style="height: 200px; width: auto; object-fit: contain; display: block;"
  />
  <img
    src="../docs/assets/net_info.png"
    alt="right image"
    style="height: 200px; width: auto; object-fit: contain; display: block;"
  />
</div>

Test hex base spin:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python test_spin.py
```


### Real-world Teleop:


> [!NOTE]
> 1. Update CAMERA_SERIAL in [constants.py](tidybot2/constants.py).
> 1. Make sure the onboard computer and the two iPhones are under the same wifi network.
> 2. Download and open the XR Browser app on your two iPhones.
> 1. Follow [XR Browser useage](https://tidybot2.github.io/docs/usage/#connecting-the-client)
> 1. The first connected iphone will control the left arm and the base xy; the second connected iphone will control the right arm and the base yaw.
> 1. Place iPhone pose before pressing start episode:
(For proper coordinate frame alignment, the phone should face the same direction as the robot when you press "Start episode".)



```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python hex_base_server.py 
```

open another tab:

```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python yam_server.py --channel can_follower_l --rpc-port 50003
```

open another tab:

```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python yam_server.py --channel can_follower_r --rpc-port 50002
```

open another tab:

```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python main.py --teleop --save
```

> [!NOTE]
> 1. If you meet this error:  
   `[2026-04-05 01:43:20.437] [host] [error] Searched, but no actual device found by given DeviceInfo: DeviceInfo(name=, deviceId=19443010B128714800, X_LINK_ANY_STATE, X_LINK_ANY_PROTOCOL, X_LINK_ANY_PLATFORM, X_LINK_SUCCESS) Check your USB connection.`
Check your USB connection.


Review collected data:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python reviewer.py
```


Sort collected data:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate
cd ~/simple-mobile/simple_mobile/tidybot2
uv run python sort_demos_from_review.py  # data/demos/review_results_20260407_171358.json
```

Next --> [diffusion policy training](../diffusion_policy/README.md)

