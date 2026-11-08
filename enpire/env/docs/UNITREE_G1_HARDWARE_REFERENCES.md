# Official Unitree G1 hardware references

References checked on 2026-10-02. Use manufacturer documents and the model
matching the actual hardware; specification tables alone do not validate a station.

| Source | Use |
| --- | --- |
| [G1 / G1 EDU specifications](https://www.unitree.com/g1/) | Dimensions, configurable degrees of freedom, sensing, optional compute, and Dex3 joint configuration. |
| [G1 SDK developer portal](https://support.unitree.com/home/en/G1_developer) | Manufacturer's development and hardware documentation; the portal requires a JavaScript-capable browser. |
| [G1 manuals and tutorials](https://www.unitree.com/app/g1/) | Official user, remote-control, battery, waist-fastener, and Dex3 assembly manuals, plus calibration/startup tutorials. |
| [Official G1 user manual](https://marketing.unitree.com/article/en/G1/User_Manual.html) | Manufacturer-hosted user-manual entry point. |
| [G1 robot description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description) | URDF/MJCF geometry, frame chain, joints, and hardware-revision/model selection table. |
| [Official teleoperation hardware guide](https://github.com/unitreerobotics/xr_teleoperate/blob/main/Device.md) | Identifies the built-in RealSense D435i and distinguishes external stereo-camera mounts. |

The reviewed
[29-body-joint Dex3 revision-1 model](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf)
defines `head_joint` and `d435_joint` as fixed joints attached to `torso_link`.
Its camera geometry is a nominal factory mounting reference. Select the model
for the actual `mode_machine` and mechanical revision before using its joint
limits or transforms. Different models contain different geometry.

For these robots, next verify whether the head D435i uses the factory mount.
If it does, the matching URDF can provide an initial mounting transform for
validation. It does not supply each camera's active-stream lens calibration,
prove an RGB optical-frame convention, establish the box pose, or confirm
camera/state timing. Read lens parameters from the existing RealSense setup
without opening a competing capture pipeline. Save all robot-specific evidence
outside Git, and retain the physical sanity check and explicit motion gate.
