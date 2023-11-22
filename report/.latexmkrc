# latexmk -pdf -pvc main.tex
$pdflatex = "pdflatex -shell-escape -synctex=1 -halt-on-error %O %S";
$sleep_time = 1; # 1 second
$pdf_mode = 1;   # tex -> pdf

# If you have your work split up into several parts, you have to specify the main file like this:
@default_files = ('report.tex');

# And finally, if latexmk -c refuses to remove certain files, you can specify their extensions and next time they’ll be gone:
$clean_ext = "acn acr alg glg glo gls ist log nav out run.xml slg snm syg syi synctex.gz dpth";

# https://tex.stackexchange.com/questions/83341/clean-bbl-files-with-latexmk-c
$bibtex_use = 2;

$out_dir = 'build';