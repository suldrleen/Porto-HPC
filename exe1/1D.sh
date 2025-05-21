#!/bin/bash

# Compiler et exécuter le programme pour les conditions aux limites périodiques
g++ -o Hperiodic1D.exe Hperiodic1D.cpp -fopenmp -larmadillo
if [ $? -ne 0 ]; then
    echo "Échec de la compilation de Hperiodic1D.cpp"
    exit 1
fi

./Hperiodic1D.exe
if [ $? -ne 0 ]; then
    echo "Échec de l'exécution de Hperiodic1D.exe"
    exit 1
fi

# Vérifier si le fichier de sortie existe et générer le graphique
if [ -f "DosPeriodic1D.txt" ]; then
    gnuplot <<-EOF
        set title 'Density of States (Periodic Boundary Conditions)'
        set ylabel 'DOS'
        set xlabel 'Energy'
        set terminal jpeg
        set output 'DosPeriodic1D.jpeg'
        plot 'DosPeriodic1D.txt' using 1:2 with lines
EOF
else
    echo "DosPeriodic1D.txt not found!"
fi

# Compiler et exécuter le programme pour les conditions aux limites ouvertes
g++ -o Hopen1D.exe Hopen1D.cpp -fopenmp -larmadillo
if [ $? -ne 0 ]; then
    echo "Échec de la compilation de Hopen1D.cpp"
    exit 1
fi

./Hopen1D.exe
if [ $? -ne 0 ]; then
    echo "Échec de l'exécution de Hopen1D.exe"
    exit 1
fi

# Vérifier si le fichier de sortie existe et générer le graphique
if [ -f "DosOpen1D.txt" ]; then
    gnuplot <<-EOF
        set title 'Density of States (Open Boundary Conditions)'
        set ylabel 'DOS'
        set xlabel 'Energy'
        set terminal jpeg
        set output 'DosOpen1D.jpeg'
        plot 'DosOpen1D.txt' using 1:2 with lines
EOF
else
    echo "DosOpen1D.txt not found!"
fi

