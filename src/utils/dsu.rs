/// Disjoint Set Union (Union-Find) data structure with path compression and union by rank
///
/// This implementation provides nearly constant time operations for:
/// - `make_set(v)`: Create a new set containing only element v
/// - `find_set(v)`: Find the representative of the set containing v
/// - `union_sets(a, b)`: Merge the sets containing a and b
///
/// Time complexity: O(α(n)) amortized per operation, where α is the inverse Ackermann function
///
/// Based on: https://cp-algorithms.com/data_structures/disjoint_set_union.html
#[derive(Debug, Clone)]
pub struct DisjointSetUnion {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSetUnion {
    /// Create a new DSU with n elements, each in its own set
    pub fn new(n: usize) -> Self {
        let dsu = Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        };
        dsu
    }

    /// Create a new set containing only element v
    /// Note: This is automatically done in `new()` for all elements 0..n
    fn make_set(&mut self, v: usize) {
        if v >= self.parent.len() {
            // Extend the DSU if needed
            let old_size = self.parent.len();
            self.parent.resize(v + 1, 0);
            self.rank.resize(v + 1, 0);
            self.size.resize(v + 1, 1);

            // Initialize new elements
            for i in old_size..=v {
                self.parent[i] = i;
                self.rank[i] = 0;
                self.size[i] = 1;
            }
        } else {
            self.parent[v] = v;
            self.rank[v] = 0;
            self.size[v] = 1;
        }
    }

    /// Find the representative of the set containing v with path compression
    fn find_set(&mut self, v: usize) -> usize {
        if v >= self.parent.len() {
            self.make_set(v);
        }

        if v == self.parent[v] {
            v
        } else {
            // Path compression: make all nodes on the path point directly to the root
            self.parent[v] = self.find_set(self.parent[v]);
            self.parent[v]
        }
    }

    /// Find the representative without modifying the structure (for immutable access)
    pub fn find_set_immutable(&self, mut v: usize) -> usize {
        if v >= self.parent.len() {
            return v; // Element doesn't exist, return itself
        }

        while v != self.parent[v] {
            v = self.parent[v];
        }
        v
    }

    /// Union two sets containing elements a and b using union by rank
    pub fn union_sets(&mut self, a: usize, b: usize) {
        let a_root = self.find_set(a);
        let b_root = self.find_set(b);

        if a_root != b_root {
            // Union by rank: attach smaller rank tree under root of higher rank tree
            if self.rank[a_root] < self.rank[b_root] {
                self.parent[a_root] = b_root;
                self.size[b_root] += self.size[a_root];
            } else if self.rank[a_root] > self.rank[b_root] {
                self.parent[b_root] = a_root;
                self.size[a_root] += self.size[b_root];
            } else {
                // Same rank: make b_root the parent and increment its rank
                self.parent[a_root] = b_root;
                self.rank[b_root] += 1;
                self.size[b_root] += self.size[a_root];
            }
        }
    }

    /// Check if two elements are in the same set
    pub fn same_set(&mut self, a: usize, b: usize) -> bool {
        self.find_set(a) == self.find_set(b)
    }

    /// Check if two elements are in the same set without modifying the structure
    pub fn same_set_immutable(&self, a: usize, b: usize) -> bool {
        self.find_set_immutable(a) == self.find_set_immutable(b)
    }

    /// Get the size of the set containing element v
    pub fn set_size(&mut self, v: usize) -> usize {
        let root = self.find_set(v);
        self.size[root]
    }

    /// Get the number of disjoint sets
    pub fn count_sets(&mut self) -> usize {
        let n = self.parent.len();
        let mut count = 0;
        for i in 0..n {
            if self.find_set(i) == i {
                count += 1;
            }
        }
        count
    }

    /// Get all elements in the same set as v
    pub fn get_set_elements(&mut self, v: usize) -> Vec<usize> {
        let root = self.find_set(v);
        let n = self.parent.len();
        let mut elements = Vec::new();

        for i in 0..n {
            if self.find_set(i) == root {
                elements.push(i);
            }
        }
        elements
    }

    /// Get all elements in the same set as v without modifying the structure
    pub fn get_set_elements_immutable(&self, v: usize) -> Vec<usize> {
        let root = self.find_set_immutable(v);
        let n = self.parent.len();
        let mut elements = Vec::new();

        for i in 0..n {
            if self.find_set_immutable(i) == root {
                elements.push(i);
            }
        }
        elements
    }

    /// Get all sets as a vector of vectors
    pub fn get_all_sets(&mut self) -> Vec<Vec<usize>> {
        use std::collections::HashMap;

        let n = self.parent.len();
        let mut sets: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..n {
            let root = self.find_set(i);
            sets.entry(root).or_insert_with(Vec::new).push(i);
        }

        sets.into_values().collect()
    }

    /// Reset the DSU to initial state (each element in its own set)
    pub fn reset(&mut self) {
        let n = self.parent.len();
        for i in 0..n {
            self.parent[i] = i;
            self.rank[i] = 0;
            self.size[i] = 1;
        }
    }

    /// Get the current capacity (number of elements the DSU can handle)
    pub fn capacity(&self) -> usize {
        self.parent.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut dsu = DisjointSetUnion::new(5);

        // Initially, each element should be in its own set
        assert_eq!(dsu.find_set(0), 0);
        assert_eq!(dsu.find_set(1), 1);
        assert!(!dsu.same_set(0, 1));

        // Union two sets
        dsu.union_sets(0, 1);
        assert!(dsu.same_set(0, 1));

        // Union with another element
        dsu.union_sets(1, 2);
        assert!(dsu.same_set(0, 2));
        assert!(dsu.same_set(1, 2));

        // Element not in the same set
        assert!(!dsu.same_set(0, 3));
        assert!(!dsu.same_set(2, 4));
    }

    #[test]
    fn test_set_size() {
        let mut dsu = DisjointSetUnion::new(5);

        assert_eq!(dsu.set_size(0), 1);

        dsu.union_sets(0, 1);
        assert_eq!(dsu.set_size(0), 2);
        assert_eq!(dsu.set_size(1), 2);

        dsu.union_sets(0, 2);
        assert_eq!(dsu.set_size(0), 3);
        assert_eq!(dsu.set_size(1), 3);
        assert_eq!(dsu.set_size(2), 3);
    }

    #[test]
    fn test_count_sets() {
        let mut dsu = DisjointSetUnion::new(5);
        assert_eq!(dsu.count_sets(), 5);

        dsu.union_sets(0, 1);
        assert_eq!(dsu.count_sets(), 4);

        dsu.union_sets(2, 3);
        assert_eq!(dsu.count_sets(), 3);

        dsu.union_sets(0, 2);
        assert_eq!(dsu.count_sets(), 2);
    }

    #[test]
    fn test_get_set_elements() {
        let mut dsu = DisjointSetUnion::new(5);

        dsu.union_sets(0, 1);
        dsu.union_sets(1, 2);

        let mut set = dsu.get_set_elements(0);
        set.sort();
        assert_eq!(set, vec![0, 1, 2]);

        let mut set = dsu.get_set_elements(3);
        set.sort();
        assert_eq!(set, vec![3]);
    }

    #[test]
    fn test_dynamic_expansion() {
        let mut dsu = DisjointSetUnion::new(3);

        // Access element beyond initial capacity
        dsu.make_set(5);
        assert_eq!(dsu.find_set(5), 5);
        assert_eq!(dsu.capacity(), 6);

        dsu.union_sets(0, 5);
        assert!(dsu.same_set(0, 5));
    }

    #[test]
    fn test_immutable_operations() {
        let mut dsu = DisjointSetUnion::new(5);
        dsu.union_sets(0, 1);
        dsu.union_sets(2, 3);

        // Test immutable find
        assert_eq!(dsu.find_set_immutable(0), dsu.find_set_immutable(1));
        assert_ne!(dsu.find_set_immutable(0), dsu.find_set_immutable(2));

        // Test immutable same_set
        assert!(dsu.same_set_immutable(0, 1));
        assert!(!dsu.same_set_immutable(0, 2));
    }

    #[test]
    fn test_reset() {
        let mut dsu = DisjointSetUnion::new(4);

        dsu.union_sets(0, 1);
        dsu.union_sets(2, 3);
        assert_eq!(dsu.count_sets(), 2);

        dsu.reset();
        assert_eq!(dsu.count_sets(), 4);
        assert!(!dsu.same_set(0, 1));
        assert!(!dsu.same_set(2, 3));
    }
}
