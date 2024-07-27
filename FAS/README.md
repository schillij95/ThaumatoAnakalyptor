Computing Feedback Arc Sets for Very Large Graphs
===

This repository contains efficient implementations for computing feedback arc sets on large graphs. The details of the implementations and their optimizations are described in the following paper: 

*Michael Simpson, Venkatesh Srinivasan, Alex Thomo:
Efficient Computation of Feedback Arc Set at Web-Scale. PVLDB 10(3): 133-144 (2016)*


Remarks
--

All algorithms load the graph in compressed form in main memory. Because of the superb compression that WebGraph provides, we can fit in the main memory of a moderate machine a large graph, such as Clueweb 2012, which has more than 42 billion edges
(http://law.di.unimi.it/webdata/clueweb12/).


Compiling
--
Compile as follows:

__javac -cp "lib/*" -d bin *.java__


Input
--

The graphs should be in WebGraph format.  

There are three files in this format: 

*basename.graph* <br>
*basename.properties* <br>
*basename.offsets*

There many available datasets in this format in:
http://law.di.unimi.it/datasets.php

Let us see for an example dataset, *cnr-2000*, in 
http://law.di.unimi.it/webdata/cnr-2000

There you can see the following files available for download.

*cnr-2000.graph* <br>
*cnr-2000.properties* <br>
*cnr-2000-t.graph* <br>
*cnr-2000-t.properties* <br>
*...* <br>
(you can ignore the rest of the files)

The first two files are for the forward (regular) *cnr-2000* graph. The other two are for the transpose (inverse) graph. If you only need the forward graph, just download: 

*cnr-2000.graph* <br>
*cnr-2000.properties*

What's missing is the "offsets" file. This can be easily created by running:

__java -cp "lib/*" it.unimi.dsi.webgraph.BVGraph -o -O -L cnr-2000__


Edgelist format
--
This section is for the case when your graph is given a text file of edges (known as edgelist). *If your graph is already in WebGraph format, skip to the next section.* 

It is very easy to convert an edgelist file into WebGraph format. 
I am making the folloiwng assumptions: 

1. The graph is unlabeled and the vertices are given by consecutive numbers, 0,1,2,... <br> (If there are some vertices "missing", e.g. you don't have a vertex 0 in your file, it's not a problem. WebGraph will create dummy vertices, e.g. 0, that does not have any neighbor.) 

2. The edgelist file is TAB separated (not comma separated). 

Now, to convert the edgelist file to WebGraph format execute the following steps: 

Sort the file, then remove any duplicate edges:

**sort -nk 1 edgelistfile | uniq > edgelistsortedfile**

(If you are on Windows, download *sort.exe* and *uniq.exe* from http://gnuwin32.sourceforge.net/packages/coreutils.htm)

Run: 

__java -cp "lib/*" it.unimi.dsi.webgraph.BVGraph -1 -g ArcListASCIIGraph dummy basename &lt; edgelistsortedfile__

(This will create *basename.graph, basename.offsets, basename.properties*.
The basename can be e.g. __simplegraph__)



Running
--

__java -cp "bin:lib/*" algorithm_name basename__

e.g. <br> java -cp "bin:lib/*" ArrayFAS simplegraph

(Change : to ; if you are on Windows)

The result will be output to the console.



Contact
--

For any questions, send email to simpsonm@uvic.ca
