/*
    FAS via Quicksort

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import java.util.BitSet;
import java.util.Arrays;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class KwikSortFAS {

	String basename; // input graph basename
   	ImmutableGraph G; // graph G
	int n; // number of vertices in G
	int[] A; // array of nodes to be sorted
 	Random rand = new Random();
		
	// load graph and initialize class variables
	public KwikSortFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		n = G.numNodes();
		System.out.println("n="+n);
		System.out.println("e="+G.numArcs());	
		
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
	
	public boolean edgeTo(int u, int w) {
     	if (Arrays.binarySearch(G.successorArray(u), w) >= 0) {
           	return true;
     	} else {
           	return false;
     	}
	}
	
	public boolean edgeTo_Linear(int u, int w) {
     	for (int s : G.successorArray(u)) {
           	if (s == w) {
                 	return true;
           	}
     	}
     	return false;
	}
	
	public void kwiksort(int[] A, int lo, int hi, PrintWriter writer) {
     	if (hi <= lo) return;
     	
     	int lt = lo, gt = hi;
     	int v = lo + rand.nextInt(hi - lo + 1);
     	int i = lo;
     	
     	while (i <= gt) {
     	
             	if      (edgeTo(A[i],A[v]) && i!=v) swap(A, lt++, i++);
           	else if (edgeTo(A[v],A[i]) && i!=v) swap(A, i, gt--);
           	else                  i++;
         }
     	
         //System.out.println("lo="+lo+" hi="+hi+" lt="+lt+" gt="+gt+" v="+v);
         kwiksort(A, lo, lt-1, writer);
         if (lt > lo || gt < hi) {
              if (gt > lt) writer.println(gt-lt);
              kwiksort(A, lt, gt, writer);
         }
         kwiksort(A, gt+1, hi, writer);
    
	}
	
	public void kwiksort_int(int[] A, int lo, int hi) {
     	
     	if (hi <= lo) return;
     	
     	int lt = lo, gt = hi;
     	int v = A[lo + rand.nextInt(hi - lo + 1)];
     	int i = lo;
     	
     	while (i <= gt) {
           	if      (A[i] < v) swap(A, lt++, i++);
           	else if (A[i] > v) swap(A, i, gt--);
             	else               i++;
     	}
     	
     	kwiksort_int(A, lo, lt-1);
     	kwiksort_int(A, gt+1, hi);
    
	}
	
	public void swap(int[] A, int i, int j) {
     	int temp = A[i];
     	A[i] = A[j];
     	A[j] = temp;
	}
	
	public long computeFAS() throws Exception {
     	// reset A
     	for(int i = 0; i < A.length; i++) {
      		A[i] = i;
		}
		shuffle(A);
		
		String outfile = basename + "_Esizes.txt";
     	PrintWriter writer = new PrintWriter(outfile, "UTF-8");
	
     	kwiksort(A, 0, n-1, writer);
     	
     	writer.close();
     	
     	int[] varray = new int[n];
		int i = 0;
		for(int u : A) {
			varray[u] = i;
			i++;
		}
     	
     	BitSet fvs = new BitSet(n);
		long fas = 0;
		int self = 0;
		
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
					fvs.set(v);
					fas++;
				}
      		}
		}
        
        //System.out.println("fvs size is " + fvs.cardinality());
        System.out.println("fas size is " + fas);
        //System.out.println("self loops = " + self);
        
        return fas;
	}
	
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
		
		KwikSortFAS fas = new KwikSortFAS(args[0]);
		
		long min_fas = Long.MAX_VALUE;
		for (int i = 0; i < 1; i++) {
      		long curr_fas = fas.computeFAS();
      		if (curr_fas < min_fas) {
            		min_fas = curr_fas;
      		}
		}
		System.out.println("min fas size is " + min_fas);
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
