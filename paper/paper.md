---
title: 'hashin\_shtrikman\_mp: a package for the optimal design and discovery of multi-phase composite materials'
tags:
  - Python
  - materials
  - composites
  - design
  - optimization
authors:
  - name: Carla J. Becker
    corresponding: true
    affiliation: 1
  - name: Hrushikesh Sahasrabuddhe
    affiliation: "2, 3"
    orcid: 0000-0001-7346-4568
  - name: Max C. Gallant
    affiliation: "2, 4"
  - name: Anubhav Jain
    affiliation: 3
    orcid: 0000-0001-5893-9967
  - name: Kristin A. Persson
    affiliation: "2, 4"
    orcid: 0000-0003-2495-5509
  - name: Tarek I. Zohdi
    affiliation: 1
    orcid: 0000-0002-0844-3573

affiliations:
 - name: Department of Mechanical Engineering, University of California, Berkeley, California, United States of America
   index: 1
   ror: 00hx57361
 - name: Department of Materials Science and Engineering, University of California, Berkeley, California, United States of America
   index: 2
 - name: Energy Technologies Area, Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA
   index: 3
 - name: Materials Sciences Division, Lawrence Berkeley National Laboratory, Berkeley, California, United States of America
   index: 3
date: 31 March 2025
bibliography: paper.bib

---

# Summary

\verb|hashin_shtrikman_mp| is a tool for composites designers who have desired composite properties in mind, but who do not yet have an underlying formulation. The library utilizes the tightest theoretical bounds on the effective properties of composite materials with unspecified microstructure – the Hashin-Shtrikman bounds – to identify candidate theoretical materials, find real materials that are close to the candidates, and determine the optimal volume fractions for each of the constituents in the resulting composite. Its i) leveraging of materials in the [Materials Project](https://next-gen.materialsproject.org/) database, ii) integration with the [Materials Project API](https://next-gen.materialsproject.org/api}{Materials Project API), iii) use of genetic machine-learning and iv) agnosticism to underlying microstructure, and ultimate engineering application, make it a tool with much broader applications than its predecessors. 

# Statement of need

Composites are ubiquitous in engineering due to their tunability and enhanced material properties as compared to their individual constituents. As such, composite design is an active field, but the pursuit of new materials through experimentation is expensive. Today, computational tools for materials design are integral to reducing the cost and increasing the pace of innovation in the energy, electronics, aviation sectors, and beyond.

Several Python packages already exist for specific areas in composites modeling, such as [CompositesLib](https://github.com/rafaelcidade/compositeslib), [Compysite](https://github.com/echaffey/Compysite), [FeCLAP](https://github.com/azzeddinetiba/FeCLAP), and [material-mechanics](https://pypi.org/project/material-mechanics/), all of which perform stress analysis on laminates and/or fiber-reinforced composites using either classical laminate theory or the finite element method. Others exist which, like \verb|hashin_shtrikman_mp|, utilize the Hashin-Shtrikman bounds on effective composite properties, such as  [BurnMan](https://geodynamics.github.io/burnman/) for thermal analysis of composite rocks/assemblages, [rockphypy](https://rockphypy.readthedocs.io/en/latest/getting_started/08_Shaly_sand_modelling.html) for mechanical modeling of sand-shale systems, \citep{ZARE2017176}'s modeling of clay nanocomposites, and \citep{ZERHOUNI2019344}'s modeling of 3D printed microstructures. All of these tools, however, are highly specific to composite microstructure, macro-geometry, and composition. More notably, they focus on analysis of already well-defined composites, rather than discovery of new materials.

\verb|hashin_shtrikman_mp| is intended for composites designers who are much earlier in their design process -- designers who are seeking out new composite formulations and who are not yet tied to a specific underlying microstructure. \verb|hashin_shtrikman_mp| defines an inverse problem wherein composite formulations which achieve a desired behavior are found by minimizing a cost function [@zohdi2012electromagnetic]. Accounting for both absolute error from the desired properties and targeting even load distribution across constituent phases, \verb|hashin_shtrikman_mp| returns candidate theoretical materials, then searches for real materials in the Materials Project database with properties close to the recommended constituents. 

# Underlying Theory

## Estimate effective composite properties with the Hashin-Shtrikman bounds

When designing composites, simple volume-weighted linear combinations of constituent material properties do not yield accurate approximations of the resulting effective composite properties. Instead, for laminates, materials designers often bound the resulting composite properties using equations from constitutive elastic theory, such as the Hill-Reuss-Voight-Weiner bounds, where the lower bound is the harmonic mean of the constituent material properties and the upper bound is the arithmetic mean [@commentaryHS]. For quasi-isotropic and quasi-homogeneous multi-phase composites with arbitrary phase geometry (a more general case), a better option is to use the Hashin-Shtrikman bounds, which provide even tighter ranges on the resulting effective properties [@hashin1962variational]. Equation \ref{eqn:gen_ineq} summarizes the Hashin-Shtrikman bounds on a generalized effective material property $y^{*}$ of an $n$-phase composite. The generalized material properties for the $n$-phases are ordered from least to greatest where $y_{1} \leq y_{2} \leq \dotsb \leq y_{n}$ with corresponding volume fractions sum to unity $v_{1} + v_{2} + \dotsb + v_{n} = 1$:

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References