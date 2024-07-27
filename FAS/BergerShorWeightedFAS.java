/*
    FAS via DFS back edges

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.util.Random;
import java.util.BitSet;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

import it.unimi.dsi.webgraph.labelling.ArcLabelledImmutableGraph;
import it.unimi.dsi.webgraph.labelling.ArcLabelledNodeIterator;
import it.unimi.dsi.webgraph.labelling.Label;

public class BergerShorWeightedFAS {

	String basename; // input graph basename
   	ArcLabelledImmutableGraph G, I; // graph G
	int n; // number of vertices in G
	long e; // number of edges in G
	int[] A; // array of nodes to be sorted
 	Random rand = new Random();
 	BitSet deleted;
		
	// load graph and initialize class variables
	public BergerShorWeightedFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ArcLabelledImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		System.out.println("Loading transpose graph...");
		I = ArcLabelledImmutableGraph.load(basename+"-t"); //We need random access
		System.out.println("Graph loaded");
		
		n = G.numNodes();
		e = G.numArcs();
		System.out.println("n="+n);
		System.out.println("e="+e);	
		
		A = new int[n];
		for(int i = 0; i < A.length; i++) {
      		A[i] = i;
		}
		
		deleted = new BitSet(n);
	}
	
	public void shuffle(int[] array) {
     	int count = array.length;
     	for (int i = count; i > 1; i--) {
           	swap(array, i - 1, rand.nextInt(i));
          }
	}
	
	public void swap(int[] A, int i, int j) {
     	int temp = A[i];
     	A[i] = A[j];
     	A[j] = temp;
	}
	
	public int computeFAS() {
	
     	shuffle(A);
     	long mag = 0;
     	int fas = 0;
     	int in;
     	int out;
     	int w;
	
     	for (int v = 0; v < A.length; v++) {
     	
           	in = 0;
           	out = 0;
     	
           	int[] in_neighbors = I.successorArray(A[v]);
     		int in_deg = I.outdegree(A[v]);
     		Label[] in_labels = I.labelArray(A[v]);
     		for(int x = 0; x < in_deg; x++) { 
      			w = in_neighbors[x];
      			if(A[v]==w)
      				continue;
      			if (!deleted.get(w))
           			in+=in_labels[x].getInt();
      		}
      		
      		int[] out_neighbors = G.successorArray(A[v]);
     		int out_deg = G.outdegree(A[v]);
     		Label[] out_labels = G.labelArray(A[v]);
     		for(int x = 0; x < out_deg; x++) { 
      			w = out_neighbors[x];
      			if(A[v]==w)
           			continue;
      			if (!deleted.get(w))
           			out+=out_labels[x].getInt();
      		}
     	
           	if (in > out) {
                 	mag += in;
                 	fas += out;
           	} else {
                 	mag += out;
                 	fas += in;
           	}
           	
      		deleted.set(A[v]);
     	}
     	
     	System.out.println("fas weight is " + (double)(fas)/100);
     	
     	return fas;
	}
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		// args = new String[] {"word_assoc_test"};
		
		if(args.length != 1) {
			System.out.println("Usage: java dfsFAS basename");
			System.out.println("Output: FAS statistics");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		BergerShorWeightedFAS fas = new BergerShorWeightedFAS(args[0]);
		fas.computeFAS();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
