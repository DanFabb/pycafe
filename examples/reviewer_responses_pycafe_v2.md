# Reviewer Responses for `pycafe_v2.tex`

## Reviewer 1

### Reviewer comments

Recommendation: Revisions Required

Please find below the observations on each segment of the meta paper

Title - The title is self-explanatory and descriptive enough. It seems okay to me.

Abstract - The current abstract is written in a "marketing-style"; revise the last few lines to be more technical and objective.

Keywords- At least 5–7 searchable, domain-relevant keywords are recommended.

Keywords should include:

Python

Software framework

(Domain-specific terms depending on PyCafe’s application)

Open-source software, etc.

Introduction- I don't find any comparison of PyCafe with existing alternatives, with proper citations. Please include that. Also, highlight what makes this tool unique from others.

Implementation and Architecture- The inclusion of an architectural diagram is okay, but it has to be extended further; workflow illustrations and example execution outputs(in separate figures) would substantially improve clarity and usability.

Quality Control- The Quality Control section requires considerable strengthening, as it does not adequately describe the testing strategy, validation methodology, reliability measures, or any automated testing and continuous integration mechanisms. A clear explanation of how correctness, robustness, and reproducibility are ensured is essential to meet research software publication standards.

Reuse Potential- The Reuse Potential section should be expanded to provide concrete scenarios in which the software can be applied, extended, or integrated with other tools. Clear guidance for developers is missing who may wish to modify or build upon the software would increase its practical value to the research community. At the repository level, documentation should be further improved by providing comprehensive installation instructions, dependency specifications, example input and output data, and reproducible usage workflows. The addition of troubleshooting guidance and structured API documentation would enhance accessibility for new users.

Figures and Diagrams- Add system architecture diagram, detailed workflow diagram, provide example output screenshots, and finally add dependency or module relationship visualisation. Without these diagrams, the paper is not complete.

License - If any third-party software/library has been used, then mention any licensing issues with the package/library

Sample Data and Reproducibility - Include an example dataset which may be used with pyCAFE and Python.

Overall Suggestion - Please conform to the above changes, it's a notable work, but providing support and maintenance through a proper support mechanism is very much required so that the scientific community can take leverage on your development or extend your work.

### Author response

- The abstract was rewritten in a more technical and objective style, with emphasis on explicit matrix assembly, supported boundary conditions, test coverage, CI, and analytical validation.
- The keywords were expanded and now include searchable terms such as `Python`, `Helmholtz equation`, `frequency domain`, `modal analysis`, `computational acoustics`, and `open source`.
- The introduction now compares pyCAFE with FEniCSx, FreeFEM, and scikit-fem, and clarifies that pyCAFE is differentiated by its matrix-explicit, acoustics-focused workflow.
- The implementation section was extended with architecture and workflow figures, plus validation and output figures showing eigenfrequencies, mode shapes, and benchmark comparisons.
- The quality-control section was substantially strengthened and now describes the pytest suite, test categories, analytical validation, convergence study, and GitHub Actions CI.
- The reuse-potential section was expanded with concrete educational, benchmarking, and research-use scenarios, plus example notebooks and extension paths.
- The manuscript now includes comparison figures/tables against FEniCSx and COMSOL, together with workflow-oriented examples in the repository.
- Licensing is clarified through the MIT license in the software availability section; third-party packages are cited in the dependencies and references.
- Reproducibility support is improved through examples, validation scripts, repository sample data, and a Zenodo archive DOI.
- Support and maintenance are addressed through the public repository workflow, issues, and contribution invitation in the reuse section.

## Reviewer 2

### Reviewer comments

Recommendation: Revisions Required

This manuscript presents pyCAFE, an open-source Python framework for solving two-dimensional frequency-domain acoustic problems using the finite element method. The software focuses on transparent assembly of stiffness, mass, and damping matrices and supports commonly used acoustic boundary conditions. The contribution is valuable from a research software accessibility and reproducibility perspective.

The architecture diagrams and workflow illustrations are clear and help readers understand the structure of the framework and analysis pipeline. The comparison examples against commercial FEM software are helpful for demonstrating numerical consistency and correctness.

Here are my comments that focusing on software engineering, usability, and reproducibility aspects :

1. Software Reproducibility and User Onboarding

The manuscript would benefit from a short “getting started” or reproducibility workflow describing how a new user can:

• Obtain the software

• Install dependencies

• Run a minimal example case

This would help new users quickly verify installation and functionality.

2. Documentation and Usage Clarity

If not already available in the repository, it may be helpful to include:

• A minimal working example script

• Expected input/output example files

• Typical workflow description from mesh → boundary → solver → post-processing

3. Validation Presentation

The validation against commercial FEM software is useful. Consider briefly clarifying:

• How test cases were selected

• Whether additional benchmark cases exist or are planned

### Author response

- A reproducibility-oriented workflow is now present through the availability section, explicit installation command (`pip install pycafe`), and repository examples.
- The repository includes minimal working examples in notebook form for both modal analysis and direct frequency sweeps.
- The manuscript now describes the typical workflow from mesh generation to boundary conditions, assembly, solver, and post-processing.
- Validation against commercial software is clarified and complemented by analytical validation, convergence data, and open-source cross-validation with FEniCSx.
- The validation section now better explains the chosen benchmark cavity and documents multiple validation paths already included in the repository.

## Reviewer 3

### Reviewer comments

Recommendation: Revisions Required

Comments for Authors

Strengths

Clear problem motivation and contribution
The manuscript clearly identifies the gap between symbolic FEM frameworks and matrix-explicit FEM workflows.
The positioning of pyCAFE as matrix-explicit and educationally transparent is strong and well justified.
The abstract clearly states that pyCAFE explicitly assembles stiffness, mass, and damping matrices and supports multiple acoustic boundary conditions.

Good modular software architecture description
The architecture explanation (core + submodules + scripts) is easy to follow.
The separation of solver, boundary condition assignment, and mesh handling is good software engineering practice.
The manuscript explains that scripts import submodules rather than embedding logic in the core, enabling flexible simulation setup.

Strong reproducibility orientation
External mesh generation via Gmsh
Explicit matrices exposed to user
Open-source code with MIT license
The software works on externally generated meshes and integrates with open scientific Python libraries.

Major Technical Improvement Suggestions

Need stronger benchmarking and validation section
Currently:

Only comparison with one commercial FEM software is mentioned.
No quantitative table of error metrics across multiple test cases.
The manuscript states validation was performed by replicating geometry and comparing natural frequencies and mode shapes.

Suggest adding:

Multiple benchmark geometries
Convergence study (mesh refinement vs error)
Frequency sweep validation
Comparison vs analytical Helmholtz solutions (where available)

Testing coverage is too limited for software journal
Currently:

Only minimal pytest functional workflow test is described.
The test suite mainly verifies package import and a minimal FEM example workflow.

Suggest:

Unit tests for:
Element matrix correctness
Boundary condition enforcement
Solver accuracy
CI pipeline integration (GitHub Actions)

Missing performance discussion
For JORS software papers, readers expect:

Runtime scaling
Memory scaling
Sparse solver performance
Add:

DOF vs runtime graph
Frequency sweep computational complexity
Comparison vs FEniCS / COMSOL / MATLAB baseline (if possible)

Limited dimensionality scope (2D only)
The manuscript focuses only on 2D frequency domain acoustics.

Suggest:

Explicit roadmap discussion:
3D FEM extension
Time domain acoustics
Vibroacoustic coupling
PML implementation details

Moderate Improvement Suggestions

Improve novelty positioning
Currently novelty = matrix-explicit + Python + educational transparency.

Strengthen by clearly stating:

What pyCAFE does that FEniCS cannot
What pyCAFE does that MATLAB CAFE could not

Add usage example workflow
Add:

Step-by-step example simulation
Code snippet
Expected output visualization

Improve software accessibility section
Add:

pip install instructions
Docker container (optional but strong)
Example datasets

Minor / Writing Suggestions

Grammar / Style

Some sentences are very long → reduce cognitive load.
Use bullet points more in Implementation section.
Consistency

Use consistent naming:
"frequency domain acoustic"
"frequency-domain acoustics"

### Author response

- We clarified that the contribution is software-oriented rather than a new FEM formulation; the novelty lies in transparency, explicit operator access, and an acoustics-focused Python workflow.
- The benchmarking section was strengthened with analytical validation, an `h`-refinement convergence study, FEniCSx comparison, and COMSOL comparison.
- The quality-control section now documents a broader automated test suite, including element matrices, boundary conditions, modal solver checks, and Helmholtz solver checks.
- Performance discussion was improved through solver times, DOF counts, and comparison plots against FEniCSx in the validation section.
- The two-dimensional scope is now stated more explicitly, and the roadmap discusses planned extensions to 3D, PMLs, vibroacoustics, and time-domain solvers.
- The reuse section now includes concrete workflow examples and explains how the notebooks can be used as templates for new studies.

## Reviewer 4

### Reviewer comments

The manuscript presents a standard frequency-domain Helmholtz finite element formulation with explicit assembly of acoustic stiffness, mass, and damping matrices, but it does not introduce any novel discretization schemes, stabilization techniques, operator reformulations, limit its methodological contribution. In addition, explicit portion-level assembly for linear and quadratic quadrilateral elements strictly follows conventional finite component procedures as described in textbooks, offering zero enhancements in computational efficiency, accuracy, and stability beyond established methods. Also, the absence an h-refinement convergence researcher using Gmsh-generated meshes prevents verification asymptotic consistency in L2 and H1 norms, leaving the reliability numerical solutions increasingly refined meshes untested. As a result, solver relies exclusively on sparse direct solvers from SciPy ambiguous implementing iterative Krylov subspace methods like GMRES and BiCGSTAB, limiting scalability, computational efficiency, and applicability to large-scale acoustic problems. Consequently, use of Guyan reduction to enforce prescribed pressure boundary conditions is missing accompanied by a truncation error analysis justification, raising concerns about accuracy, stability, and the correctness borderline condition implementation. Despite this, Perfectly Matched Layers (PMLs) are mentioned in the roadmap as a potential feature, they are flawed actually implemented in the current framework, which prevents the simulation from accurately representing open-domain acoustic wave propagation. For this reason, prescribed normal velocity boundary implementation is included in the formulation deficient been verified against one-dimensional finite element benchmarks, making it unclear whether the boundary integration is computed accurately. Moreover, software architecture separates core computation, mesh loading, and solver modules but lacks formal object-oriented abstraction documentation, making it difficult other developers to extend, maintain, and adapt the code alternative FEM applications. Nevertheless, No comparison with established open source finite element frameworks such as FEniCS, FreeFEM, Elmer FEM, MFEM is provided which prevents the manuscript from situating its methodology within the context of widely used standards and best practices. Finally, post-processing utilities are restricted to basic Matplotlib visualization without support advanced field export formats, data interoperability, integration with industrial post-processing pipelines, reducing practical utility for complex acoustic simulations.

### Author response

- The manuscript now explicitly states that pyCAFE does not claim a new discretization scheme; the contribution is an open, matrix-based research software implementation.
- An `h`-refinement convergence study is now included for the rectangular cavity, with quantitative error tables and reference slopes.
- The direct and modal solvers are validated against analytical and independent numerical references, reducing the concern that results rely on a single benchmark only.
- Prescribed-pressure enforcement is now documented as a partitioned Guyan-style reduction, with equations showing how known and unknown DOFs are separated.
- The current solver scope is described transparently: sparse direct solution for frequency sweeps and ARPACK-based eigensolution for modal analysis.
- PMLs are presented only as future work in the roadmap, not as an implemented feature.
- Comparison against open-source software is now included through FEniCSx, while commercial comparison is reported against COMSOL.
- Post-processing support is broader than basic Matplotlib only: the manuscript now documents point-response extraction, animation, VTU export, and ParaView compatibility.

## Reviewer 5

### Reviewer comments

Proposed pycafe framework absences methodological novelty as it reproduces classical frequency-domain acoustic FEM formulations with explicit matrix assembly procedures already widely implemented in commercial solvers and mature open-source numerical simulation libraries.

Validation is restricted to a single modal comparison with a commercial solver, lacking analytical benchmark verification using canonical problems such as rectangular cavity eigenmode solutions tube analytical reference models.

Restriction to two dimensional quadrilateral elements significantly limits engineering applicability and geometric flexibility while, the paper provides neither theoretical a methodological roadmap for, extending implementation toward triangular or hybrid meshes.

The absence of comparison with existing Python-based acoustic FEM frameworks prevents objective evaluation computational efficiency, numerical accuracy, extensibility, and software usability, making the framework’s practical advantages unclear.

Adoption Guyan reduction prescribed pressure enforcement omits theoretical justification and comparative performance analysis against alternative constraint enforcement techniques including Lagrange multipliers, penalty formulations, variationally consistent Nitsche-based approaches.

Quality control section explicitly acknowledges limited verification and testing procedures, which is insufficient scientific software publications requiring rigorous validation regression testing, verification, and automated numerical consistency assessment.

Framework fails implementation advanced non-reflecting boundary conditions beyond simple impedance models, excluding sophisticated inventions such as perfectly matched layers, infinite elements, high-order radiation boundary approximations.

Solver employment missing incorporates preconditioning strategies essential for solving large-scale Helmholtz systems efficiently thereby potentially causing severe convergence degradation and computational inefficiency in high-frequency large-domain simulations.

Implementation does not address floating-point numerical conditioning challenges in complex-valued frequency-domain arithmetic risking accumulation of round-off errors and unreliable pressure predictions at high discretization density.

Framework excludes hybrid FEM–BEM coupling open-boundary formulations, restricting its capability to accurately model exterior acoustic radiation, scattering phenomena, and practical environmental engineering applications.

### Author response

- We clarified that pyCAFE does not aim to introduce methodological novelty in acoustics FEM; its contribution is openness, inspectability, and research/teaching usability.
- Validation is no longer limited to one comparison: the paper now includes analytical cavity benchmarks, convergence results, FEniCSx cross-validation, and COMSOL comparison.
- The current scope remains two-dimensional quadrilateral acoustics, and this limitation is stated clearly together with a roadmap for future element and 3D extensions.
- Comparison with an open-source Python-oriented alternative is now included through FEniCSx and discussed in the introduction.
- The Dirichlet treatment is now described explicitly with the partitioned system used for prescribed pressures.
- The quality-control section was expanded with automated tests and CI, addressing the previous concern about limited verification.
- Advanced open-domain treatments such as PML and hybrid formulations are not claimed as present capabilities; they are listed as future extensions.
- Solver scalability limitations are acknowledged, and the roadmap leaves room for future iterative/preconditioned strategies.

## Reviewer 6

### Reviewer comments

Recommendation: Revisions Required

The Paper:

Is the title of the paper descriptive and objective?

Yes

Comments (optional):

Does the Abstract give an indication of the software's functionality, and where it would be used?

Yes

Comments (optional)

Clearly describes functionality (2D acoustic FEM, frequency domain, boundary conditions) and context (ported from MATLAB).

Do the keywords enable a reader to search for the software?

Yes

Comments (optional)

Acoustics; FEM; Python” are very broad. Consider adding: “frequency domain”, “Helmholtz equation”, “open source”, “modal analysis”.

Does the Introduction give enough background information to understand the context of the software's development and use?

Yes

Comments (optional)

Provides adequate context on FEM for acoustics, distinguishes from weak-form frameworks, and motivates the Python port.

Does the Implementation and Architecture section give enough information to get an idea of how the software is designed, and any constraints that may be placed on its use?

Yes

Comments (optional)

Well-explained modular structure with clear diagrams (Figure 1). The separation into 7 workflow steps is clearly presented.

Does the Quality Control section adequately explain how the software results can be trusted?

Yes

Comments (optional)

Mentions pytest-based tests and validation against commercial software.

Does the Reuse section provide concrete and useful suggestions for reuse of the software, for instance: other potential applications, ways of extending or modifying the software, integration with other software?

Yes

Comments (optional):

Clear use cases (cavities, ducts), extensibility discussed, and roadmap provided (Figure 4).

Are figures and diagrams used to enhance the description? Are they clear and meaningful?

Yes

Comments (optional):

Figures 1-4 are clear and enhance understanding. Figure 1 shows the architecture, Figure 2 provides validation data, Figure 3 demonstrates typical output, and Figure 4 presents the development roadmap. Minor issue: “Heigth” should be “Height” in Figure 2.

Do you believe that another researcher could take the software and use it, or take the software and build on it?

Yes

Comments (optional):

The modular architecture, clear documentation on ReadTheDocs, examples directory, and MIT license all facilitate reuse and extension.

The software:

Is the software in a suitable repository? (see our recommended repositories as listed on the journal's About page for more information)

Yes

Does the software have a suitable open licence? (see our FAQ for data papers on the journal's About page for more information)

Yes

Comments (optional):

MIT License is an appropriate open-source license that permits broad reuse.

If the Archive section is filled out, is the link in the form of a persistent identifier, e.g. a DOI? Can you download the software from this link?

No

Comments (optional):

The paper lists the GitHub URL as “Persistent identifier” but GitHub URLs are not persistent identifiers. The authors should archive a release on Zenodo (or similar service) and provide a DOI for long-term archival and citation purposes.

If the Code repository section is filled out, does the identifier link to the appropriate place to download the source code? Can you download the source code from this link?

Yes

Is the software license included in the software in the repository? Is it included in the source code?

Yes

Comments (optional):

LICENSE file is present in the repository root.

Is sample input and output data provided with the software?

Yes

Comments (optional):

An examples directory exists with mesh files and demonstration scripts.

Is the code adequately documented? Can a reader understand how to build/deploy/install/run the software, and identify whether the software is operating as expected?

Yes

Comments (optional):

Documentation is available on ReadTheDocs (pycafe.readthedocs.io) with installation guide, API reference, and getting started guide. The README also provides workflow examples.

Does the software run on the systems specified? (if you do not have access to a system with the prerequisite requirements, let us know)

Yes

Comments (optional):

The software runs on Linux with Python 3.11+ and the specified dependencies.

Is it obvious what the support mechanisms for the software are?

Yes

Comments (optional):

GitHub issues are available for support, and the paper mentions contacting the corresponding author.

Summary comments to the author(s):

Please provide a list of your recommendations, indicating which are compulsory for acceptance

Persistent Identifier: The GitHub URL listed as “Persistent identifier” in the Availability section is not a persistent identifier. Please archive a release on Zenodo (or similar) and provide a DOI. This is essential for long-term citation and reproducibility.

Commercial software identification: Please specify which commercial finite element software was used for the validation comparison in Figure 2 and the Quality Control section. This is important for reproducibility and transparency.

Typo correction: “Heigth” should be “Height” in Figure 2.

Please list any comments that are optional but would improve the quality or the reusability of the software:

Keywords: Consider adding more specific keywords such as “Helmholtz equation”, “modal analysis”, “frequency response” to improve discoverability.

Comparison with existing tools: A brief comparison or differentiation from existing Python acoustic/FEM tools (e.g., scikit-fem, FEniCS acoustic applications) would strengthen the positioning.

Installation command: Including a brief mention of the installation command (e.g., pip install or installation from source) in the paper would be helpful.

Reference [6]: This reference appears to have missing author names.

PyPI submission: Consider submitting the package to PyPI for easier installation.

Analytical validation: For simple geometries (e.g., rectangular cavities), analytical solutions exist and could provide additional validation evidence.

### Author response

- The manuscript now includes more specific and searchable keywords, including `Helmholtz equation`, `frequency domain`, and `modal analysis`.
- The commercial validation software is now identified explicitly as COMSOL Multiphysics 6.2 in the manuscript.
- The software archive requirement is addressed through a Zenodo DOI in the availability section.
- The validation has been expanded with analytical cavity benchmarks and convergence evidence, in line with the reviewer recommendation.
- The installation command is now stated explicitly in the availability section.
- The manuscript still needs one final proofreading pass for small figure/text issues such as the reported typo corrections and any remaining placeholder text.

## Reviewer 7

### Reviewer comments

Recommendation: Revisions Required

The Paper:

Is the title of the paper descriptive and objective?

Yes

Comments (optional):

Does the Abstract give an indication of the software's functionality, and where it would be used?

Yes

Comments (optional)

Do the keywords enable a reader to search for the software?

Yes

Comments (optional)

Does the Introduction give enough background information to understand the context of the software's development and use?

Yes

Comments (optional)

The reviewer welcomes comparison against commercial FEM software where appropriate. However, there is no discussion on similar open-source solutions (even if non-existent, this could be mentioned, or the reach of other python packages in the field of acoustics can be highlighted, so the contribution of this package can be suited in the research area).

Does the Implementation and Architecture section give enough information to get an idea of how the software is designed, and any constraints that may be placed on its use?

Yes

Comments (optional)

Does the Quality Control section adequately explain how the software results can be trusted?

Yes

Comments (optional)

Does the Reuse section provide concrete and useful suggestions for reuse of the software, for instance: other potential applications, ways of extending or modifying the software, integration with other software?

Yes

Comments (optional):

Are figures and diagrams used to enhance the description? Are they clear and meaningful?

Yes

Comments (optional):

Do you believe that another researcher could take the software and use it, or take the software and build on it?

Yes

Comments (optional):

The software:

Is the software in a suitable repository? (see our recommended repositories as listed on the journal's About page for more information)

Yes

Does the software have a suitable open licence? (see our FAQ for data papers on the journal's About page for more information)

Yes

Comments (optional):

If the Archive section is filled out, is the link in the form of a persistent identifier, e.g. a DOI? Can you download the software from this link?

No

Comments (optional):

Not filled out.

If the Code repository section is filled out, does the identifier link to the appropriate place to download the source code? Can you download the source code from this link?

Yes

Is the software license included in the software in the repository? Is it included in the source code?

Yes

Comments (optional):

Is sample input and output data provided with the software?

Yes

Comments (optional):

Is the code adequately documented? Can a reader understand how to build/deploy/install/run the software, and identify whether the software is operating as expected?

Yes

Comments (optional):

Code documentation would benefit from additional comments in the example or full Read the Docs page; however, the code itself is appropriately documented.

Does the software run on the systems specified? (if you do not have access to a system with the prerequisite requirements, let us know)

Yes

Comments (optional):

Is it obvious what the support mechanisms for the software are?

Yes

Comments (optional):

Summary comments to the author(s):

Please provide a list of your recommendations, indicating which are compulsory for acceptance

In the following, I attach my review of the pyCAFE: A Finite Element Framework for Solving Acoustic Problems in Python. The package is available at the https://github.com/DanFabb/pycafe/tree/main and can be easily reviewed, downloaded, and installed. The information on the package homepage matches the information in the submitted manuscript. After reviewing these, I was able to perform a review of the contribution, which is elaborated in the following.
pyCAFE is an open-source solver for two-dimensional acoustic problems based on the finite element method. It operates on an externally provided two-dimensional mesh, supplied by open-source tools. Within the package workflow, boundary conditions and material properties are assigned, followed by assembly of global system matrices (mass, stiffness, and damping matrices) for the given acoustic problem. Two analysis types are available within the package, namely modal and direct analysis. The package offers an option to employ post processing visualization tools provided with the package, with emphasis on sound pressure visualization over the computational domain.
The paper title, abstract and keywords are meaningful to the field of research. The content of the contribution is thus clear, while the abstract already implies important (from the user's perspective) limitations of the package (e.g. limited to two dimensional acoustic problems). The introduction elaborates the need for the open-source package, with the references on the theoretical background provided at meaningful discussions. The reviewer welcomes comparison against commercial FEM software where appropriate. However, there is no discussion on similar open-source solutions (even if non-existent, this could be mentioned, or the reach of other python packages in the field of acoustics can be highlighted, so the contribution of this package can be suited in the research area). The introduction concludes with an introduction of pyCAFE and the goal of open-source package development. To improve the readability of the Introduction, the reviewer has the following comments, that refer to the theoretical background on the package rather than the open-source package itself:
- »In these applications, frequency domain formulations are widely adopted, particularly when time harmonic responses are of interest.« Why are frequency domain formulations preferred?
- What is low to mid frequency range?
- Sentence »While some open source tools can solve Helmholtz type problems, many frameworks adopt a variational weak form workflow in which the governing equations are expressed symbolically and assembled implicitly.« carries a lot of information and is challenging to understand.
Implementation and architecture are clearly presented, therefore the workflow using the pyCAFE package is clear. The reviewer only has the following comments:
- When discussing geometry supported by pyCAFE, the limitation to two-dimensional problems could be highlighted again.
- It is unclear what the authors mean by »The software provides auxiliary scripts for geometry creation to ensure consistent naming conventions and physical group definitions to support this process.«.
- The authors only consider the example that adopts an externally provided mesh. Can the package be used to define boundary conditions and perform a solution given that the mass, stiffness, and damping matrices are available as inputs?
Within the section Geometry and material selection, it is slightly unclear when the authors discuss the support of simple shaped problems: rectangular and circular cavities. Does that mean given that the more complex 2D geometry is analyzed using external meshing solutions, pyCAFE is not able to solve anything beyond rectangular/circular shape? Potential typo in »Mesh generation is performed through the Gmsh Python API [16] addition to creating the geometric entities, the script assigns physical groups to both the domain and each boundary segment.«.
In the Build matrices section ,the following sentences are not perfectly clear:
- »The build matrices folder contains the numerical part useful for the construction of the acoustic finite element operators.«
- »By following a fully matrix based approach, the resulting operators are directly accessible to the user, enabling inspection, debugging, and validation against reference solutions.«
Sections on boundary conditions, solver and post processing are well elaborated and clear to the reviewer.
Quality control description is adequate. Proposed test is simplistic, but can be easily adopted by new code developers. In the https://github.com/DanFabb/pycafe/blob/main/tests/test_basic.py, there is an emoji in the comments. Although not discouraged by PEP8, is that on purpose?
Reuse potential is well elaborated. Fig. 3 is hard to read; the reviewer would prefer that the font on the figure is made slightly bigger. To-do timeline is reasonable. The authors already identified shortcomings of the current package state (in the reviewer's eyes, especially a Read the Docs page is desirable).
Package usage is possible as the example is provided on the repository. However, the example would significantly benefit from connecting text between the cells, instead of just running cells of code, especially since a significant amount of arguments needs to be provided within each step of the example.
The package is deposited in a suitable repository and has a suitable license. In the paper, the link to the repository is provided to download the source code. Sample input and output data are provided (also needed to run example and test). Code documentation would benefit from additional comments in the example or full Read the Docs page; however, the code itself is appropriately documented. Software did run on the system specified in the paper.
The package is a useful contribution to the field of vibroacoustics. The paper is easy to read and understand, which is welcomed by the reviewer. The authors could also discuss alternative packages in the field, and where their shortcomings lie, i.e. what is the novel contribution in the field by this very work.
All points considered, I see the paper in the scope of the journal and a nice contribution to both open-source and vibroacoustic community. All my comments considered, I suggest the paper to undergo minor revision before being published.

### Author response

- The introduction now better situates pyCAFE with respect to open-source alternatives such as FEniCSx, FreeFEM, and scikit-fem.
- The two-dimensional scope of pyCAFE is stated clearly in the manuscript and revisited in the reuse/future-work discussion.
- The workflow wording around geometry support was clarified by separating simple auxiliary geometry scripts from the general externally generated mesh workflow.
- The implementation, matrix-building, and boundary-condition sections were expanded so that the package responsibilities are easier to follow.
- The example-based reuse discussion was strengthened: the two notebooks are now described explicitly as modal and direct-sweep templates.
- The reviewer’s point about example readability is addressed at repository level by improving notebook documentation and adding a validation notebook/workflow.
- The overall positioning as a useful open-source vibroacoustics contribution is preserved, while the revised manuscript now documents validation and reuse more concretely.

## Final checks still needed before submission

- Replace any remaining placeholders in `pycafe_v2.tex`, especially if any text still refers to temporary notes.
- Do one final language-editing pass for grammar, spacing, and consistency of terms such as `frequency-domain acoustics`.
- Verify that all cited figures are present with readable fonts and corrected labels.
- Ensure the repository and Zenodo release match the manuscript version exactly.

## Final checks still needed before submission
- Replace any remaining placeholders in `pycafe_v2.tex`, especially if any text still refers to temporary notes.
- Do one final language-editing pass for grammar, spacing, and consistency of terms such as `frequency-domain acoustics`.
- Verify that all cited figures are present with readable fonts and corrected labels.
- Ensure the repository and Zenodo release match the manuscript version exactly.
