(ns mnist-classifier.util
  (:require [clojure.java.io :as io]
            [clojure-csv.core :as csv]
            [clojure.core.matrix :as m]))

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

(defn get-rand
  "Returns a random value between -epsilon and +epsilon."
  [epsilon]
  (- (* (rand) 2 epsilon) epsilon))

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


(defn class-vector
  "Turns each element i in y into a zero-filled vector with element i set to 1."
  [y]
  (map #(m/set-row (m/zero-vector 10) % 1) y))

(defn sigmoid-prime
  "Calculates the derivative of the logistic function for every element."
  [m]
  (m/emul (m/logistic m) (m/sub 1 (m/logistic m))))

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
  (let [m (m/row-count X)
        [theta1 theta2] weights
        a1 (m/join-along 1 (one-matrix m 1) X)
        z2 (m/mmul a1 (m/transpose theta1))
        a2 (m/join-along 1 (one-matrix m 1) (m/logistic z2))
        z3 (m/mmul a2 (m/transpose theta2))
        a3 (m/logistic z3)]
    (mapv index-of a3 (mapv m/emax a3))))

(defn now
  []
  (.getTime (new java.util.Date)))

