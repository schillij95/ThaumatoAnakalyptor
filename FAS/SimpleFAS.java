/*
    FAS via DFS back edges

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.util.BitSet;
import java.util.Random;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class SimpleFAS {

	String basename; // input graph basename
   	ImmutableGraph G; // graph G
	int n; // number of vertices in G
	long e; // number of edges in G
	int[] A; // array of nodes to be sorted
 	Random rand = new Random();
		
	// load graph and initialize class variables
	public SimpleFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		n = G.numNodes();
		e = G.numArcs();
		System.out.println("n="+n);
		System.out.println("e="+e);	
		
		A = new int[n];
		for(int i = 0; i < A.length; i++) {
      		A[i] = i;
		}
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
	
     	shuffle(A);
	
     	int[] varray = new int[n];
		int i = 0;
		for(int u : A) {
			varray[u] = i;
			i++;
		}
     	
		long mag = 0;
		int self = 0;
		long fas = 0;
		
		NodeIterator vi = G.nodeIterator();
		while (vi.hasNext()) {
			int v = vi.next();
			
			int[] v_neighbors = G.successorArray(v);
			int v_deg = G.outdegree(v);
			
			for(int x = 0; x < v_deg; x++) { 
				int w = v_neighbors[x];
				
				if(v==w) { // Self-loop, ignore
      				self++;
					continue;
				}
				
				if (varray[v] > varray[w]) {
					mag++;
				}
      		}
		}
		
		if (mag > e/2) {
            	fas = e - mag;
      	} else {
          	fas = mag;
      	}
        
        //System.out.println("fvs size is " + fvs.cardinality());
        System.out.println("fas size is " + fas);
        //System.out.println("self loops = " + self);
        
        return fas;
	}
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		//args = new String[] {"cnr-2000"};
		//args = new String[] {"wordassociation-2011"};
		
		if(args.length != 1) {
			System.out.println("Usage: java dfsFAS basename");
			System.out.println("Output: FAS statistics");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		SimpleFAS fas = new SimpleFAS(args[0]);
		fas.computeFAS();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
