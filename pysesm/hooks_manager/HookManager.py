import logging
import wandb
from typing import Dict, Optional
from pysesm.enums.HookTypeEnum import HookType

class HookManager:
    """
    Manages hooks for the ISTALayer and DictLayer, allowing users to activate hooks
    and specify whether data should be stored locally or logged to Weights & Biases (W&B).

    Attributes:
        active_hooks (Dict[HookType, bool]): Tracks which hooks are active.
        use_wandb (bool): Whether to log hook data to W&B.
        local_storage (Dict[HookType, list]): Stores hook data locally if not using W&B.
        logger (logging.Logger): Logger for debugging and information.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        project_name: str =None,
        active_hooks: Optional[list[HookType]] = None,  # List of hooks to activate
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the HookManager.

        Args:
            use_wandb (bool): Whether to log hook data to W&B. Defaults to False.
            active_hooks (list[HookType], optional): List of hooks to activate. Defaults to None.
            logger (logging.Logger, optional): Logger for debugging and information.
        """
        self.active_hooks = {
            HookType.ISTALAYER: False,
            HookType.DICTLAYER: False,
        }
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.local_storage = {
            HookType.ISTALAYER: [],
            HookType.DICTLAYER: [],
        }
        self.logger = logger or logging.getLogger(__name__)

        # Initialize W&B if use_wandb is True
        if self.use_wandb:
            try:
                wandb.init(project=self.project_name)  # Initialize W&B with the provided project name
                self.logger.info(f"W&B initialized with project: {self.project_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize W&B: {e}")
                self.use_wandb = False  # Disable W&B if initialization fails

        # Activate hooks if specified
        if active_hooks:
            self.activate_hooks(active_hooks)

    def activate_hooks(self, hook_types: list[HookType]) -> None:
        """
        Activates hooks for the specified layers.

        Args:
            hook_types (list[HookType]): List of hook types to activate.
        """
        for hook_type in hook_types:
            self.active_hooks[hook_type] = True
            if self.logger:
                self.logger.info(f"Hook for {hook_type.name} activated.")

    def deactivate_hooks(self, hook_types: list[HookType]) -> None:
        """
        Deactivates hooks for the specified layers.

        Args:
            hook_types (list[HookType]): List of hook types to deactivate.
        """
        for hook_type in hook_types:
            self.active_hooks[hook_type] = False
            if self.logger:
                self.logger.info(f"Hook for {hook_type.name} deactivated.")

    def log_hook_data(self, hook_type: HookType, data: Dict) -> None:
        """
        Logs hook data either locally or to W&B.

        Args:
            hook_type (HookType): Type of the hook.
            data (Dict): Data to log, typically containing parameters, gradients, and losses.
        """
        if not self.active_hooks.get(hook_type, False):
            return

        if self.use_wandb:
            try:
                # Check if W&B is initialized before logging
                if wandb.run is not None:
                    wandb.log(data)
                else:
                    self.logger.warning("W&B is not initialized. Skipping logging.")
            except Exception as e:
                self.logger.error(f"Failed to log data to W&B: {e}")
        else:
            # Store data locally
            self.local_storage[hook_type].append(data)

        if self.logger:
            self.logger.debug(f"Logged data for {hook_type.name}: {data}")

    def get_local_storage(self, hook_type: HookType) -> list:
        """
        Retrieves locally stored data for a specific hook.

        Args:
            hook_type (HookType): Type of the hook.

        Returns:
            list: Locally stored data for the hook.
        """
        return self.local_storage.get(hook_type, [])