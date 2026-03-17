The user wants to set the status of a plan folder to "complete".

The way we do that is to move the parent folder from the `_tasks/_in-progress/` folder to the `_tasks/_complete/` folder. It's possible the feature is still in the `_tasks/_planning/` folder or some other folder too, so check around the `_/dev-tasks/` subfolders.

For example, the user may provide you with either an implementation guide doc like `_tasks/_in-progress/2026-02-12_telegram-integration/2026-02-12_implementation-guide.md` or perhaps the parent folder like `_tasks/_in-progress/2026-02-12_telegram-integration/` or perhaps even just a description of the feature/initiative. Move the entire feature folder from its current location to `_tasks/_complete/`.

When you're done, please commit the changes.
