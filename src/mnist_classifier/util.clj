(ns mnist-classifier.util
  (:require [clojure.java.io :as io]
            [clojure-csv.core :as csv]
            [clojure.core.matrix :as m])
  (:import org.ejml.simple.SimpleMatrix))

(defn to-int-array
  "Converts an array of strings to an array of integers."
  [string-array]
  (map #(Integer/parseInt %) string-array))

(defn read-csv
  "Loads data from csv file."
  [filename]
  (with-open [file (io/reader filename)]
    (csv/parse-csv (slurp file))))

(defn extract-labels
  "Extracts labels from csv data. Returns a vector of integers."
  [data]
  (mapv #(Integer/parseInt (first %)) data))

(defn extract-features
  "Extracts features from csv data. Returns a vector of integers."
  [data]
  (mapv to-int-array (map #(drop 1 %) data)))

(defn normalise
  "Normalises each value in matrix and returns new matrix.
  Note: only works for matrices with positive values."
  [sm]
  (let [maximum (.elementMaxAbs sm)
        minimum (.elementMinAbs sm)]
    (.divide sm (- maximum minimum))))

(defn get-rand
  "Returns a random value between -epsilon and +epsilon."
  [epsilon]
  (- (* (rand) 2 epsilon) epsilon))

(defn round
  "Rounds to `precision` significant figures."
  [n precision]
  (let [factor (Math/pow 10. precision)]
    (/ (Math/floor (* n factor)) factor)))

(defn init-weights
  "Initialise weights for layer.
  Returns a `num-outputs` x `num-inputs + 1` matrix."
  [num-inputs, num-outputs]
  (let [epsilon (/ (Math/sqrt 6) (Math/sqrt (+ num-inputs num-outputs)))
        matrix (m/new-matrix num-outputs (inc num-inputs))]
    (m/emap #(+ % (get-rand epsilon)) matrix)))

(defn one-matrix
  "Returns a matrix of given dimensions, filled with ones"
  [x y]
  (m/emap inc (m/zero-matrix x y)))

(defn one-sm-matrix
  "Returns a SimpleMatrix of given dimensions, filled with ones"
  [x y]
  (let [ones (SimpleMatrix. x y)]
    (.fill ones 1)
    ones))

(defn to-simple-matrix
  "Converts a core.matrix matrix/vector to an EJML SimpleMatrix.
  If c-matrix is vector, the SimpleMatrix returned will be a 1 x n matrix."
  [c-matrix]
  (def s-matrix
    (if (m/matrix? c-matrix)
       (SimpleMatrix. (m/row-count c-matrix)
                      (m/column-count c-matrix))
       (SimpleMatrix. 1
                      (m/row-count c-matrix))))

  (if (m/matrix? c-matrix)
    (doseq [r (range (.numRows s-matrix))
            c (range (.numCols s-matrix))]
      (.set s-matrix r c (m/select c-matrix r c)))

    (doseq [c (range (.numCols s-matrix))]
      (.set s-matrix 0 c (m/select c-matrix c 0))))

  s-matrix)

(defn to-core-matrix
  "Converts an EJML SimpleMatrix to a core.matrix matrix."
  [s-matrix]
  (def c-matrix (m/zero-matrix (.numRows s-matrix)
                               (.numCols s-matrix)))

  (reduce (fn [mat [x y]]
           (m/mset mat x y (.get s-matrix x y)))
        c-matrix
        (for [x (range (.numRows s-matrix))
              y (range (.numCols s-matrix))]
             [x y])))

(defn drop-first-column
  "Returns a new SimpleMatrix without the first column."
  [sm]
  (.cols sm 1 SimpleMatrix/END))

(defn take-first-column
  "Returns a new SimpleMatrix made from the first column of the input matrix."
  [sm]
  (.cols sm 0 1))

(defn class-vector
  "Turns each integer i in vector y into a zero-filled vector with element i
  set to 1 so that 3 becomes [0 0 0 1 0 0 0 0 0 0 0 ]"
  [y]
  (map #(m/set-row (m/zero-vector 10) % 1) y))

(defn sigmoid
  "Calculates logistic function for every element in SimpleMatrix"
  [m]
  (let [exp (.elementExp m)]
    (.elementDiv exp (.plus exp 1.0))))

(defn sigmoid-prime
  "Calculates the derivative of the logistic function for every element."
  [m]
  (let [sig (sigmoid m)
        ones (.createLike m)]
    (.fill ones 1)
    (.elementMult sig (.minus ones sig))))

(defn index-of
  [array element]
  (.indexOf array element))

(defn predict
  "Returns the predicted digit [0-9] for a given feature vector x."
  [x weights]
  (let [[theta1 theta2] weights
        a1 (m/join [1] x)
        z2 (m/mmul a1 (m/transpose theta1))
        a2 (m/join [1] (m/logistic z2))
        z3 (m/mmul a2 (m/transpose theta2))
        a3 (m/logistic z3)]
        (index-of a3 (m/emax a3))))

(defn predict-all
  "Same as `predict` but for a matrix of features.
  Returns a vector containing a prediction [0-9] for each example."
  [X weights]
  (let [m (count (first X))
        [theta1 theta2] weights
        a1 (m/join-along 1 (one-matrix m 1) X)
        z2 (m/mmul a1 (m/transpose theta1))
        a2 (m/join-along 1 (one-matrix m 1) (m/logistic z2))
        z3 (m/mmul a2 (m/transpose theta2))
        a3 (m/logistic z3)]
    (mapv index-of a3 (mapv m/emax a3))))

(defn display
  "Display the training example in the console."
  [m]
  (doseq [y (range (.numCols m))]
    (if (and (= (rem y 28) 0) (not= y 0)) (println))
    (if (> (.get m 0 y) 0.5) (print "o") (print "."))))

(defn now
  []
  (.getTime (new java.util.Date)))

