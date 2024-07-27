/*
    FAS via DFS back edges

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.BitSet;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class BergerShorFAS {

	String basename; // input graph basename
   	ImmutableGraph G, I; // graph G
	int n; // number of vertices in G
	long e; // number of edges in G
	int[] A; // array of nodes to be sorted
 	Random rand = new Random();
 	BitSet deleted;
		
	// load graph and initialize class variables
	public BergerShorFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		System.out.println("Loading transpose graph...");
		I = ImmutableGraph.load(basename+"-t"); //We need random access
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
	
	public long computeFAS() throws Exception {
	
     	String outfile = basename + "_skews.txt";
     	PrintWriter writer = new PrintWriter(outfile, "UTF-8");
	
     	shuffle(A);
     	long mag = 0;
     	int in;
     	int out;
     	int w;
	
     	for (int v = 0; v < A.length; v++) {
     	
           	in = 0;
           	out = 0;
     	
           	int[] in_neighbors = I.successorArray(A[v]);
     		int in_deg = I.outdegree(A[v]);
     		for(int x = 0; x < in_deg; x++) { 
      			w = in_neighbors[x];
      			if (!deleted.get(w)) {
           			in++;
      			}
      		}
      		
      		int[] out_neighbors = G.successorArray(A[v]);
     		int out_deg = G.outdegree(A[v]);
     		for(int x = 0; x < out_deg; x++) { 
      			w = out_neighbors[x];
      			if (!deleted.get(w)) {
           			out++;
      			}
      		}
     	
              if (in+out > 0) {
               	if (in > out) {
                     	mag += in;
                   	writer.println((double)(out) / (in+out));
               	} else {
                     	mag += out;
                     	writer.println((double)(in) / (in+out));
               	}
              }
           	
      		deleted.set(A[v]);
     	}
     	
     	writer.close();
     	
     	long fas = e - mag;
     	
     	System.out.println("fas size is " + fas);
     	
     	return fas;
	}
	
	/*
	METHOD UNNECESSARILY USES HASH SETS
	public int computeFAS() throws Exception {
	
     	shuffle(A);
     	int mag = 0;
     	int in;
     	int out;
     	int w;
	
     	for (int v = 0; v < A.length; v++) {
     	
           	in = 0;
           	out = 0;
     	
           	int[] in_neighbors = I.successorArray(v);
     		int in_deg = G.indegree(v);
     		for(int x = 0; x < in_deg; x++) { 
      			w = in_neighbors[x];
      			if (!deleted.contains(new Pair(w,v))) {
           			in++;
      			}
      		}
      		
      		int[] out_neighbors = G.successorArray(v);
     		int out_deg = G.outdegree(v);
     		for(int x = 0; x < out_deg; x++) { 
      			w = out_neighbors[x];
      			if (!deleted.contains(new Pair(v,w))) {
           			out++;
      			}
      		}
     	
           	if (in > out) {
                 	mag += in;
           	} else {
                 	mag += out;
           	}
           	
     		for(int x = 0; x < out_deg; x++) { 
      			w = out_neighbors[x];
      			deleted.add(new Pair(v,w));
      		}
     		for(int x = 0; x < in_deg; x++) { 
      			w = in_neighbors[x];
      			deleted.add(new Pair(w,v));
      		}
     	}
     	
     	return e - mag;
	} */
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		//args = new String[] {"cnr-2000"};
		// args = new String[] {"wordassociation-2011"};
		
		if(args.length != 1) {
			System.out.println("Usage: java dfsFAS basename");
			System.out.println("Output: FAS statistics");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		BergerShorFAS fas = new BergerShorFAS(args[0]);
		fas.computeFAS();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
