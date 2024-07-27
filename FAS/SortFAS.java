/*
    FAS via Insertion Sort

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.util.Random;
import java.util.BitSet;
import java.util.Arrays;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class SortFAS {

	String basename; // input graph basename
   	ImmutableGraph G; // graph G
	int n; // number of vertices in G
	int[] A; // array of nodes to be sorted
 	Random rand = new Random();
		
	// load graph and initialize class variables
	public SortFAS(String basename) throws Exception {
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
	
	// assume 0 back edges initially
	// each time we see an edge from curr to j we decrement i.e. lose a back edge by swapping
	// each time we see an edge from j to curr we increment i.e. gain a back edge by swapping
	// looking for valley, or min value, in sequence to locate swap pos
	public void sort(int[] A) {
         
         // main loop over each element of A
         for (int i = 1; i < A.length; i++) {
              //System.out.println("Processing node " + i);
              int curr = A[i];
              int val = 0;
              int min = 0;
              int loc = i;
              
              // check all candidate positions from i to 0
              for (int j = i-1; j >= 0; j--) {
              
                  if (edgeTo(curr,A[j])) {
                      val--;
                  } else if (edgeTo(A[j],curr)) {
                      val++;
                  }
                  
                  if (val <= min) {
                      min = val;
                      loc = j;
                  }
                  
              }
              
              // shift over values and insert
              for (int t = i-1; t >= loc; t--) {
                  A[t+1] = A[t];
              }
              A[loc] = curr;
              
         }
	}
	
	// METHOD CURRENTLY FLAWED - DO NOT USE
	// val increment/decrement is mirrored compared to above appraoch
	// i.e. we increment when we see acurrent back edge and decrement when we see a current forward edge
	public void sort_flawed(int[] A) {
         
         // main loop over each element of A
         for (int i = 1; i < A.length; i++) {
              //System.out.println("Processing node " + i);
              int curr = A[i];
              int val = 0;
              
              // check all candidate positions from i to 0
              for (int j = i-1; j >= 0; j--) {
              
                  if (edgeTo(curr,A[j])) {
                      val++;
                  } else if (edgeTo(A[j],curr)) {
                      val--;
                  }
                  
              }
              
              //System.out.println("val = " + val);
              if (val == 0) {
                  // insert at pos 0 and shift
                  int k = i-1;
               	while (k >= 0) {
                   	A[k+1] = A[k];
                   	k--;
               	}
               	A[k+1] = curr;
              } else if (val < 0) {
                  // go from pos i to first num < 0 and shift
                  int k = i-1;
               	while (!edgeTo(A[k],curr)) {
                   	A[k+1] = A[k];
                   	k--;
               	}
               	A[k+1] = curr;
              } else { // val > 0
                  // go from pos 0 to first num >= 0 and shift
                  int t = 0;
               	while (edgeTo(A[t],curr)) {
                   	t++;
               	}
               	
               	int k = i-1;
               	while (k >= t) {
                   	A[k+1] = A[k];
                   	k--;
               	}
               	A[k+1] = curr;
              }
              
         }
	}
	
	public void sort_ncubed(int[] A) {
         
         // main loop over each element of A
         for (int i = 1; i < A.length; i++) {
              System.out.println("Processing node " + i);
              int curr = A[i];
              int least_back_edges_induced = Integer.MAX_VALUE;
              int best_loc = -1;
              
              // check all candidate positions from 0 to i
              for (int j = 0; j <= i; j++) {
                  int count = 0;
                  // count edges from curr node to first j-1 nodes
                  for (int k = 0; k < j; k++) {
                      if (edgeTo(curr,A[k])) {
                          count++;
                      }
                  }
                  // count edges from nodes j+1 --> i to curr node
                  for (int k = j+1; k <= i; k++) {
                      if (edgeTo(A[k],curr)) {
                          count++;
                      }
                  }
                  
                  if (count < least_back_edges_induced) {
                      best_loc = j;
                  }
              }
              
              // shift over values and insert
              for (int t = i-1; t >= best_loc; t--) {
                  A[t+1] = A[t];
              }
              A[best_loc] = curr;
         }
	}
	
	public void sort_int(int[] A) {
         
         for (int i = 1; i < A.length; i++) {
           	int j = i-1;
           	int x = A[i];
           	while (j >= 0 && A[j] > x) {
               	A[j+1] = A[j];
               	j--;
           	}
           	A[j+1] = x;
         }
    
	}
	
	public void swap(int[] A, int i, int j) {
     	int temp = A[i];
     	A[i] = A[j];
     	A[j] = temp;
	}
	
	public long computeFAS() throws Exception {
	
     	sort(A);
     	
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
		
		SortFAS fas = new SortFAS(args[0]);
		fas.computeFAS();
		
		/*
		long old_fas;
		long new_fas = Long.MAX_VALUE;
		int c = 0;
		do {
      		old_fas = new_fas;
      		new_fas = fas.computeFAS();
      		c++;
      		System.out.println("old_fas= " + old_fas + "\tnew_fas= " + new_fas + "\tc= " + c);
		} while (new_fas < old_fas);
		System.out.println("Number of iterations until convergence: " + c);
		*/
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
