docker run --rm -it -v /home/jvsguerra/remote-repos/UNICAMP/MO433/proposal/:/home adnrv/texlive latexmk -f -pdf -pdflatex=pdflatex --shell-escape %O %S final-project-proposal.tex
