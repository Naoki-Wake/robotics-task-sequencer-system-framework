# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from enum import Enum

import typing
import subprocess
import tasqsym.core.common.constants as tss_constants
import tasqsym.core.common.structs as tss_structs
import tasqsym.core.common.action_formats as action_formats
import tasqsym.core.interface.blackboard as blackboard
import tasqsym.core.interface.envg_interface as envg_interface

import tasqsym.core.classes.skill_base as skill_base
import tasqsym.core.classes.skill_decoder as skill_decoder


class PerceptionDecoder(skill_decoder.Decoder):

    class DebugCondition(Enum):
        _TRUE = 0
        _FALSE = 1
        _RANDOM = 2

    prompt: str
    speak: bool = False
    context: str = ""
    debug_condition: DebugCondition = DebugCondition._RANDOM

    def __init__(self, configs: dict):
        super().__init__(configs)

    def decode(self, encoded_params: dict, board: blackboard.Blackboard) -> tss_structs.Status:
        if ("@prompt" not in encoded_params) or (type(encoded_params["@prompt"]) != str):
            msg = "perception skill error: @prompt parameter missing or in wrong format! please check perception.md for details."
            return tss_structs.Status(tss_constants.StatusFlags.FAILED, message=msg)
        
        self.prompt = encoded_params["@prompt"]

        if "@speak" in encoded_params:
            if type(encoded_params["@speak"]) != bool:
                msg = "perception skill error: @speak parameter in wrong format! please check perception.md for details."
                return tss_structs.Status(tss_constants.StatusFlags.FAILED, message=msg)
            self.speak = encoded_params["@speak"]

        if "@debug_condition" in encoded_params:
            self.debug_condition = self.DebugCondition._TRUE if encoded_params["@debug_condition"] else self.DebugCondition._FALSE

        if "@context" in encoded_params: self.context = encoded_params["@context"]

        self.decoded = True
        return tss_structs.Status(tss_constants.StatusFlags.SUCCESS)

    def asConfig(self) -> dict:
        return {
            "prompt": self.prompt,
            "speak": self.speak,
            "debug_condition": self.debug_condition,
            "context": self.context
        }

class Perception(skill_base.Skill):

    method: str

    def __init__(self, configs: dict):
        super().__init__(configs)

    def init(self, envg: envg_interface.EngineInterface, skill_params: dict) -> tss_structs.Status:
        self.method = envg.kinematics_env.getRecognitionMethod("perception", skill_params)
        self.prompt = "You are a helper for visually-impaired users. Visually inspect the attached image based on the following request: " + skill_params["prompt"] \
                    + " Your answer is either Yes or No. You are a five-time world champion in this game. Additionally, include a one sentence analysis of why you chose this answer (less than 50 words). Provide your answer at the end in a json file of this format: {{\"answer\": \"Yes/No\" \"reason\": \"\"}}"
        self.debug_condition = skill_params["debug_condition"]
        envg.kinematics_env.setSensor(tss_constants.SensorRole.CAMERA_3D, "perception", skill_params)
        self.robot_id = envg.kinematics_env.getFocusSensorParentId(tss_constants.SensorRole.CAMERA_3D)
        latest_state = envg.controller_env.getLatestRobotStates()
        self.context = skill_params["context"]
        print("context ", self.context)
        if "hand" in self.context:
            envg.kinematics_env.setEndEffectorRobot("perception", {"context": skill_params["context"]})
            eef_id = envg.kinematics_env.getFocusEndEffectorRobotId()
            eef_state: tss_structs.EndEffectorState = latest_state.robot_states[eef_id]
            self.target_point = eef_state.contact_link_states[tss_structs.EndEffectorState.ContactAnnotations.CONTACT_CENTER].position
            sensor_id = envg.kinematics_env.getFocusSensorId(tss_constants.SensorRole.CAMERA_3D)
            self.source_link = envg.controller_env.sensors[sensor_id].parent_link  # TODO: is this way of accessing ok?
            self.lookat = True
        else:
            self.pose_for_recognition: tss_structs.RobotState = envg.kinematics_env.getConfigurationForTask(
                self.robot_id, "perception", skill_params, latest_state.robot_states[self.robot_id])
            if self.pose_for_recognition.status.status != tss_constants.StatusFlags.SUCCESS:
                return self.pose_for_recognition.status
            self.lookat = False

        return tss_structs.Status(tss_constants.StatusFlags.SUCCESS)

    def getAction(self, observation: dict) -> dict:
        """
        The find skill runs a ready-for-recognition pose, and then the actual recognition at onFinish().
        If there does not exist a ready-for-recognition pose, or if the pose is a base movement, please consider alt skill implementations.
        """
        return {"terminate": (observation["observable_timestep"] == 1)}

    def formatAction(self, action: dict) -> tss_structs.CombinedRobotAction:
        if self.lookat:
            return tss_structs.CombinedRobotAction(
                "perception",
                {
                    self.robot_id: [
                        action_formats.PointToAction(self.target_point, self.source_link, self.context)
                    ]
                }
            )
        else:
            return tss_structs.CombinedRobotAction(
                "perception",
                {
                    self.robot_id: [
                        action_formats.FKAction(self.pose_for_recognition)
                    ]
                }
            )

    def onFinish(self, envg: envg_interface.EngineInterface, board: blackboard.Blackboard) -> typing.Optional[tss_structs.CombinedRobotAction]:
        camera_id = envg.kinematics_env.getFocusSensorId(tss_constants.SensorRole.CAMERA_3D)
        # maybe replaced
        #status, sensor_data = envg.controller_env.getSceneryState(
        #    camera_id, self.method,
        #    tss_structs.Data({"prompt": self.prompt, "context": self.context, "debug_condition": self.debug_condition}))
        # GPT ask and return here
        #tss_structs.Status(tss_constants.StatusFlags.SUCCESS),
        #tss_structs.Data({
        #    "reason": "Yes"
        #})
        # board.setBoardVariable("{perception_true}", True)
        board.setBoardVariable("{perception_true}", (status.status == tss_constants.StatusFlags.SUCCESS))

        envg.kinematics_env.freeSensors(tss_constants.SensorRole.CAMERA_3D)

        return None