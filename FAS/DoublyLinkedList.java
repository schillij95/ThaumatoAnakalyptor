/******************************************************************************
 *
 *  A list implemented with a doubly linked list. 
 *
 ******************************************************************************/

public class DoublyLinkedList {
    private Node pre;     // sentinel before first item
    private Node post;    // sentinel after last item

    public DoublyLinkedList() {
        pre  = new Node();
        post = new Node();
        pre.next = post;
        post.prev = pre;
    }

    public boolean isEmpty()    { return pre.next.equals(post); }

    // add the item to the list
    public void add(Integer item) {
        Node last = post.prev;
        Node x = new Node(item);
        x.next = post;
        x.prev = last;
        post.prev = x;
        last.next = x;
    }
    
    // add the node to the list
    public void add(Node x) {
        Node last = post.prev;
        x.next = post;
        x.prev = last;
        post.prev = x;
        last.next = x;
    }

    // pop off the last node
    public Node remove() { 
        Node last = post.prev;
        Node secondLast = last.prev;
        secondLast.next = post;
        post.prev = secondLast;
        return last;
    }

}