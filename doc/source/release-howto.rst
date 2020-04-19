:orphan:

.. _release-howto:

===============
 Release HOWTO
===============

*This document is intended for PyStan developers.*

How to make a release
=====================

- Verify that the correct version is shown in ``pyproject.toml`` and ``stan/__init__.py``.
- Update ``CHANGELOG.rst``. Create Pull Request. Wait for PR to be merged.
- Tag (with signature): ``git tag -u CB808C34B3BFFD03EFD2751597A78E5BFA431C9A -s 1.2.3``, replacing ``1.2.3`` with the version.
- Push the new tag to the repository: ``git push origin 1.2.3``, replacing ``origin`` with the remote and ``1.2.3`` with the version.
- Publish (universal) wheel using ``poetry``. [TODO: add details]
