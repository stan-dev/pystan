======================
Contributing to PyStan
======================

Introduction
------------

PyStan is a Python interface to C++ functions in the `Stan library`_.
Specifically, PyStan provides an interface to the C++ functions in the ``stan::services`` namespace.

Stability and maintainability are two overriding goals of this software package.
New features which make this software easier to maintain are welcome.
Contributions which make the software more difficult to maintain are not welcome at this time.

PyStan adopts virtually all the conventions and procedures discussed in Astropy's `How to make a
code contribution`_.

In general, contributions which follow the project's coding style, have tests, and solve a specific
problem will be merged. Here the project follows the spirit of the `Collective Code Construction
Contract (C4)`_.

.. _Stan library: https://mc-stan.org
.. _How to make a code contribution: http://docs.astropy.org/en/stable/development/workflow/development_workflow.html
.. _Collective Code Construction Contract (C4): https://rfc.zeromq.org/spec:42/C4/

Coding Style
------------

PyStan code is `PEP 8`_ compliant. The project uses the code formatter black_ with a maximum
line-length of 119 characters. Documentation, comments, and docstrings should be wrapped at 79 characters, even though PEP 8 suggests 72.

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _black: https://pypi.org/project/black/

PyStan code is also checked by ``flake8`` and ``mypy``.

Commit Messages
---------------

git commit messages must be formatted carefully in order to allow the automatic generation of release notes.

You should also use the first line of your commit message to indicate the commit's "type". If it
is a bugfix, the commit message should start with "fix:". If it is a new feature, the commit
message should start with "feat:". This information makes reviewing patches and generating
release notes easier. For a full list of common commit "types", consult the `Conventional Commits` specification.

Commit messages must have a body.
The body should elaborate on the summary.
It should also explain why the change is valuable.
Our working hypothesis is that requiring commit message bodies will make it more difficult to inadvertently introduce code which makes maintaining the software more difficult.

In general, follow the `Astropy guidelines for git`_. Important reminders:

* Make frequent commits, and always include a commit message. Each commit
  should represent one logical set of changes.
* Never merge changes from ``httpstan/main`` into your feature branch.

.. _Conventional Commits: https://www.conventionalcommits.org/en/v1.0.0-beta.4/#summary
.. _Astropy guidelines for git: https://astropy.readthedocs.io/en/latest/development/workflow/development_workflow.html#astropy-guidelines-for-git

Contributor Covenant Code of Conduct
====================================

Our Pledge
----------

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our
project and our community a harassment-free experience for everyone,
regardless of age, body size, disability, ethnicity, sex
characteristics, gender identity and expression, level of experience,
education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

Our Standards
-------------

Examples of behavior that contributes to creating a positive environment
include:

-  Using welcoming and inclusive language
-  Being respectful of differing viewpoints and experiences
-  Gracefully accepting constructive criticism
-  Focusing on what is best for the community
-  Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

-  The use of sexualized language or imagery and unwelcome sexual
   attention or advances
-  Trolling, insulting/derogatory comments, and personal or political
   attacks
-  Public or private harassment
-  Publishing others’ private information, such as a physical or
   electronic address, without explicit permission
-  Other conduct which could reasonably be considered inappropriate in a
   professional setting

Our Responsibilities
--------------------

Project maintainers are responsible for clarifying the standards of
acceptable behavior and are expected to take appropriate and fair
corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit,
or reject comments, commits, code, wiki edits, issues, and other
contributions that are not aligned to this Code of Conduct, or to ban
temporarily or permanently any contributor for other behaviors that they
deem inappropriate, threatening, offensive, or harmful.

Scope
-----

This Code of Conduct applies both within project spaces and in public
spaces when an individual is representing the project or its community.
Examples of representing a project or community include using an
official project e-mail address, posting via an official social media
account, or acting as an appointed representative at an online or
offline event. Representation of a project may be further defined and
clarified by project maintainers.

Enforcement
-----------

Instances of abusive, harassing, or otherwise unacceptable behavior may
be reported by contacting the project team at riddella@indiana.edu.
Alternatively, you may use the NumFOCUS Code of Conduct reporting tool
available at https://numfocus.org/code-of-conduct.
All complaints will be reviewed and investigated and will result in a
response that is deemed necessary and appropriate to the circumstances.
The project team is obligated to maintain confidentiality with regard to
the reporter of an incident. Further details of specific enforcement
policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in
good faith may face temporary or permanent repercussions as determined
by other members of the project’s leadership.

Attribution
-----------

This Code of Conduct is adapted from the `Contributor
Covenant <https://www.contributor-covenant.org>`__, version 1.4,
available at
https://www.contributor-covenant.org/version/1/4/code-of-conduct.html

For answers to common questions about this code of conduct, see
https://www.contributor-covenant.org/faq
