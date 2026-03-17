The user wants to set the status of a plan folder to "in progress".

The way we do that is to move the parent folder from the current directory (e.g. `_tasks/_planning/` or `_tasks/_ready-to-start`) to the `_tasks/_in-progress/` folder.

For example, the user may provide you with either a feature description doc like `_tasks/_planning/2026-02-12_telegram-integration/2026-02-12_feature-description.md` or perhaps an implementation guide or perhaps the parent folder like `_tasks/_ready-to-start/2026-02-12_telegram-integration/` or perhaps even just a description of the feature/initiative in which case you should look for the folder where the feature docs live. Move the entire feature folder from its current location to `_tasks/_in-progress/`.

When you're done, please commit the changes.
