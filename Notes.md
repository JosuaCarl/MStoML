- Classification
	- Viele Metabolite (fast alle) nicht in AGORA SBML Modell
		- kaum Zeit für weitere Kuration
	- Visualisierung mit Cobra / Escher
		- Medium mGAM Metabolitzusammensetzung ? 
- VAE
	- Ergebnisse in MAE & Klassifikation schlecht
		- 0 Baseline (MSE)
		- scaling (cosine)
			- Zusätzlicher Schritt wäre kein Problem
		- Fourier transform sieht ganz anders aus -> Erkennt gedachte Verteilungsfrequenz nicht <- Uniforme Verteilung
	- 825000 bins > 1000 samples
	- MSE + Cosine Error -> Bessere convergence in Hyperparameter search
- Sonstiges
    - Andere Arten der Dimensionsreduktion
		- Letzter Arbeitsmonat für Übertragung + Anpassung des Matlab-Skripts für Untargeted peak picking
			- Centroiding & Peak Picking mit OpenMS
    - Biologische Interpretation erst mal droppen
		- Gradient Boosting für Co-occurrence
- Weiteres Vorgehen
	- Comm8 Training -> Comm20 Testing
		- Übertragbarkeit auf größere Kolonien
	- VAE mit mehr FIA Daten + Neuem Loss
	- Gegenüberstellung der 5 Algorithmen
		- Figures !
	- Paper von Linklab für Figures
	- Graphische Bearbeitung von Figures (SVGs)
	- Latent space als Spektrum zeigen ?


1. Abschluss des HP Tunings für alle Loss-Functions (je 500 Konfigurationen)
2. Training des VAE mit den neuen Daten, unter Ausschluss der COM8 Daten
3. Anwendung des VAE auf COM8 und Extraktion des latent space
4. COM8 (equal concentration combinations) community prediction mit latent space und annotierten Metaboliten
5. Training der besten Classification Algorithmen mit COM8 (equal concentration combinations)
6. Anwendung der Classification-Modelle auf COM8 (grown together) um Übertragbarkeit zu testen



ANOVA
- LDA - ANOVA Connection
- Volcano plot for all organisms + mark all picked by XGBoost
- Violin + Importance plot of all organisms