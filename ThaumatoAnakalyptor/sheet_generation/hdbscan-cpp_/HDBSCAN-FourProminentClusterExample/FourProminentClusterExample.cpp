#include<iostream>
#include<cstdio>
#include"../HDBSCAN-CPP/Hdbscan/hdbscan.hpp"
using namespace std;


int main() {
	Hdbscan hdbscan("HDBSCANDataset/FourProminentClusterDataset.csv");
	hdbscan.loadCsv(2);
	vector<vector <double>> dataset = hdbscan.dataset;
	hdbscan.execute(5, 5, "Euclidean");
	hdbscan.displayResult();
	cout << "You can access other fields like cluster labels, membership probabilities and outlier scores."<<endl;
	/*Use it like this
	hdbscan.labels_;
	hdbscan.membershipProbabilities_;
	hdbscan.outlierScores_;
	*/

	return 0;
}