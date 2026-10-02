"""
A simple file for drawing morphisms and expansions for visual analysis.
Drawn morphisms are made into a LaTeX file with the diagrams implemented in TikZ. One has to compile the .tex files separately.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import shutil
import subprocess

from sage.all import AA, QQ, PolynomialRing, latex

from formats import require
from leaves import double_leaf, expand_double_leaf, light_leaf
from morphisms import Morphism, operation_target
from reconstruct import Expansion
from quadratic import encode_scalar


def tex_text(value):
    """Escape arbitrary public generator labels and captions as TeX text."""
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$",
                    "%": r"\%", "&": r"\&", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    # Arbitrary Unicode remains portable even with minimal pdfLaTeX fonts.
    # The U+ notation is an exact character identifier, not executable TeX.
    return "".join(replacements.get(c, c) if ord(c) < 128 else r"\texttt{U+"+f"{ord(c):04X}"+"}"
                   for c in str(value))


def word_tex(word):
    return r"\varnothing" if not word else r"\,".join(r"\text{"+tex_text(s)+"}" for s in word)


def polynomial_tex(system, polynomial):
    """Render exact coefficients with public-color root names."""
    polynomial = system.ring(polynomial)
    if not polynomial:
        return "0"
    parts = []
    for powers, coefficient in sorted(polynomial.dict().items(), reverse=True):
        monomial = "".join(r"\alpha_{\text{"+tex_text(s)+"}}"+("^{"+str(e)+"}" if e != 1 else "")
                           for s, e in zip(system.labels, powers) if e)
        if system.field is AA:
            try:
                c = str(latex(QQ(coefficient)))
            except (TypeError, ValueError):
                encoded = encode_scalar(coefficient, AA)
                minimal = PolynomialRing(QQ, "t")([QQ(v) for v in encoded["minimal_polynomial"]])
                c = r"\operatorname{Root}_{"+str(encoded["real_root_index"]+1)+r"}\!\left("+str(latex(minimal))+r"\right)"
        else:
            c = str(latex(coefficient))
        if monomial and coefficient == 1:
            parts.append(monomial)
        elif monomial and coefficient == -1:
            parts.append("-"+monomial)
        elif monomial:
            parts.append(r"\left("+c+r"\right)"+monomial)
        else:
            parts.append(c)
    return " + ".join(parts).replace(" + -", " - ")


def _xy(x, y):
    return f"({x:.4f},{y:.4f})"


def _strand(x0, y0, x1, y1, color, arm=None):
    height = y1-y0
    if arm == "in":
        controls = ((x0, y0+.35*height), (.6*x0+.4*x1, y0+.6*height))
    elif arm == "out":
        controls = ((.4*x0+.6*x1, y0+.4*height), (x1, y0+.65*height))
    else:
        controls = ((x0, y0+.36*height), (x1, y1-.36*height))
    return (rf"\draw[ewstrand,{color}] {_xy(x0,y0)} .. controls {_xy(*controls[0])} and "
            rf"{_xy(*controls[1])} .. {_xy(x1,y1)};")


def tikz_morphism(morphism, *, annotate=False):
    """Return a TikZ picture of a morphism, read from bottom to top.

    Polynomial boxes are labeled p_step in their actual region; their values
    are printed by the atlas below the picture. Mixed-color intersections are
    precisely the intended 2m-valent generators. Only univalent vertices get
    filled dots. TeX comments record every operation and intermediate boundary.
    """
    if morphism.kind == "double_leaf":
        morphism = expand_double_leaf(morphism)
    require(morphism.kind == "Elias-Williamson graph", "tikz_morphism expects a generator morphism.")
    system, word, y, spacing = morphism.system, morphism.source, 0.0, .55
    color = lambda s: "ewcolor"+str(system._positions[s])
    position = lambda n, j: spacing*(j-(n-1)/2)
    out = [r"\begin{tikzpicture}[baseline=(current bounding box.center)]"]

    def boundary(current, level, anchor):
        if not current:
            out.append(rf"\node[ewletter,anchor={anchor}] at {_xy(0,level)} {{$\varnothing$}};")
        for j, s in enumerate(current):
            out.append(rf"\node[ewletter,text={color(s)},anchor={anchor}] at {_xy(position(len(current),j),level)}"
                       + " {$"+word_tex((s,))+"$};")

    boundary(word, -.04, "north")
    for j, s in enumerate(word):
        out.append(_strand(position(len(word), j), y, position(len(word), j), y+.22, color(s)))
    y += .22
    for number, operation in enumerate(morphism.payload, 1):
        after, m, k = operation_target(system, word, operation)
        i, kind = operation.position, operation.kind
        height = max(.56, .17*m) if kind == "braid" else .52
        top, middle = y+height, y+height/2
        center = spacing*(i+(m-1)/2-(len(word)-1)/2)
        out.append("% step "+str(number)+": "+json.dumps(operation.to_data(), ensure_ascii=True))
        out.append("% boundary "+json.dumps(list(word), ensure_ascii=True)+" -> "+json.dumps(list(after), ensure_ascii=True))
        for j, s in enumerate(word):
            if i <= j < i+m:
                continue
            next_j = j if j < i else j+k-m
            out.append(_strand(position(len(word), j), y, position(len(after), next_j), top, color(s)))
        if kind in ("braid", "merge", "split"):
            for j in range(i, i+m):
                out.append(_strand(position(len(word), j), y, center, middle, color(word[j]), "in"))
            for j in range(i, i+k):
                out.append(_strand(center, middle, position(len(after), j), top, color(after[j]), "out"))
        elif kind == "upper dot":
            out.append(_strand(position(len(word), i), y, center, middle, color(word[i])))
            out.append(rf"\fill[{color(word[i])}] {_xy(center,middle)} circle[radius=1.75pt];")
        elif kind == "lower dot":
            out.append(_strand(center, middle, position(len(after), i), top, color(after[i])))
            out.append(rf"\fill[{color(after[i])}] {_xy(center,middle)} circle[radius=1.75pt];")
        elif kind in ("cup", "cap"):
            current, level, direction = (word, y, 1) if kind == "cap" else (after, top, -1)
            x0, x1, h = position(len(current), i), position(len(current), i+1), .34*direction
            out.append(rf"\draw[ewstrand,{color(current[i])}] {_xy(x0,level)} .. controls "
                       rf"{_xy(x0,level+h)} and {_xy(x1,level+h)} .. {_xy(x1,level)};")
        elif kind == "polynomial":
            out.append(rf"\node[draw,fill=white,inner sep=1pt,font=\tiny] at {_xy(center,middle)} {{$p_{{{number}}}$}};")
        if annotate:
            right = spacing*max(len(word), len(after))/2+.18
            out.append(rf"\node[font=\tiny,text=gray,anchor=west] at {_xy(right,middle)} {{{number}}};")
        word, y = after, top
    if not morphism.payload:
        for j, s in enumerate(word):
            out.append(_strand(position(len(word), j), y, position(len(word), j), y+.8, color(s)))
        y += .8
    for j, s in enumerate(word):
        out.append(_strand(position(len(word), j), y, position(len(word), j), y+.22, color(s)))
    boundary(word, y+.26, "south")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def _preamble(system, title):
    out = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=16mm]{geometry}",
           r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
           r"\usepackage{amsmath,amssymb,tikz,adjustbox,longtable,booktabs}",
           r"\usepackage[colorlinks=true,linkcolor=blue]{hyperref}",
           r"\tikzset{ewstrand/.style={line width=.85pt,line cap=round,line join=round},ewletter/.style={font=\scriptsize,inner sep=1pt}}",
           r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{5pt}"]
    palette = ["C43136", "2268B0", "16804A"]
    for i in range(system.rank):
        rgb = colorsys.hsv_to_rgb((.07+.618*i) % 1, .7, .7)
        code = palette[i] if i < 3 else "".join(f"{round(v*255):02X}" for v in rgb)
        out.append(rf"\definecolor{{ewcolor{i}}}{{HTML}}{{{code}}}")
    out += [r"\begin{document}", r"{\Large\bfseries "+tex_text(title)+r"}\par",
            r"A coefficient multiplies on the left. "
            ]
    if system.field is AA:
        out.append(r"Here $\operatorname{Root}_j(p)$ is the $j$-th distinct real root of $p(t)$ in increasing order. "
                   r"Algebraic coefficients are specified exactly by this notation.\par")
    return out


def _picture(morphism, annotate, height=19):
    out = [r"\begin{center}\begin{adjustbox}{max width=.96\linewidth,max height="+str(height)+"cm}",
           tikz_morphism(morphism, annotate=annotate), r"\end{adjustbox}\end{center}"]
    for number, op in enumerate(morphism.payload, 1):
        if op.kind == "polynomial":
            out.append(rf"\(p_{{{number}}}="+polynomial_tex(morphism.system, op.polynomial)+rf"\), in region {op.position}.\par")
    return out


def render_expansion(expansion, path, *, morphism=None, annotate=False):
    selected = list(enumerate(expansion.terms, 1))
    system = expansion.system
    out = _preamble(system, "Coefficients of morphism in the double leaves basis")
    out += [r"\(F=\sum_j c_j \mathbb{L}\mathbb{L}_j\), where \(\mathbb{L}\mathbb{L}_j=\overline{\mathrm{LL}}_{\underline{\underline{y}},\mathbf{f}_j} \circ\mathrm{LL}_{\underline{\underline{x}},\mathbf{e}_j}\) where",
            r"\[\underline{x}="+word_tex(expansion.source)+r"\quad\text{and}\qquad \underline{y}="+word_tex(expansion.target)+r".\]\par"]
    if not selected:
        out.append(r"The displayed sum is \(0\).")

    out += [r"\begin{longtable}{r p{.46\linewidth} r r}", r"\toprule Term & Coefficient & Source mask & Target mask \\ \midrule\endhead"]
    for j, term in selected:
        out.append(f"{j} & $"+polynomial_tex(system, term.coefficient)+f"$ & {term.source_mask} & {term.target_mask}"+r" \\")
    out += [r"\bottomrule\end{longtable}"]
    if morphism:
        out += [r"\clearpage", rf"$F = $\par"]
        out.extend(_picture(morphism, annotate))
    for j, term in selected:
        mor = expand_double_leaf(double_leaf(system, expansion.source, expansion.target, *term.key))
        out += [r"\clearpage", rf"{{\large\bfseries Term {j}}}\par",
                r"\(c_{"+str(j)+"}="+polynomial_tex(system, term.coefficient)+r"\).\par",
                rf"Source mask {term.source_mask}; target mask {term.target_mask}.\par"]
        out.extend(_picture(mor, annotate))
    out.append(r"\end{document}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out)+"\n", encoding="utf-8")
    return path


def render_morphism(morphism, path, *, annotate=False):
    """Render any expression graph as named formulas and generator pictures.

    Shared DAG nodes appear once. Tensor and composition formulas refer to
    those nodes; they are not flattened into an exponentially large sum.
    A root expansion delegates to the coefficient atlas.
    """
    if morphism.kind == "expansion":
        return render_expansion(morphism.payload, path, annotate=annotate)
    nodes, seen = [], {}

    def visit(value):
        if value in seen:
            return seen[value]
        for child in value.children:
            visit(child)
        seen[value] = len(nodes)
        nodes.append(value)
        return seen[value]

    root = visit(morphism)
    out = _preamble(morphism.system, "Elias--Williamson morphism")
    out.append(rf"The requested morphism is \(F=N_{{{root}}}\). Shared expressions are named below.\par")
    for number, node in enumerate(nodes):
        out += [r"\clearpage", rf"{{\large\bfseries Node {number}}}\par",
                rf"\(N_{{{number}}}:B_{{"+word_tex(node.source)+r"}\to B_{"+word_tex(node.target)+r"}\).\par"]
        refs = [f"N_{{{seen[c]}}}" for c in node.children]
        if node.kind in ("Elias-Williamson graph", "double_leaf"):
            morphism = expand_double_leaf(node) if node.kind == "double_leaf" else node
            out.extend(_picture(morphism, annotate))
        elif node.kind == "zero":
            out.append(r"\[0\]")
        elif node.kind == "sum":
            out.append(r"\["+" + ".join(refs)+r"\]")
        elif node.kind == "scale":
            out.append(r"\[\left("+polynomial_tex(node.system, node.payload)+r"\right)"+refs[0]+r"\]")
        elif node.kind == "compose":
            out.append(r"\["+refs[1]+r"\circ "+refs[0]+r"\]")
        elif node.kind == "tensor":
            out.append(r"\["+refs[0]+r"\otimes "+refs[1]+r"\]")
        elif node.kind == "expansion":
            out.append(r"\[\sum_j c_j\mathrm{LL}_{b,f_j}^{*}\mathrm{LL}_{a,e_j}\]")
            out += [r"\begin{longtable}{r p{.46\linewidth} r r}", r"Term & Coefficient & Source mask & Target mask \\"]
            for j, term in enumerate(node.payload.terms, 1):
                out.append(f"{j} & $"+polynomial_tex(node.system, term.coefficient)+f"$ & {term.source_mask} & {term.target_mask}"+r" \\")
            out.append(r"\end{longtable}")
    out.append(r"\end{document}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out)+"\n", encoding="utf-8")
    return path