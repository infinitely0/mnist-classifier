(ns mnist-classifier.train
  (:use [mnist-classifier.util]
        [mnist-classifier.cost])
  (:require [clojure.core.matrix :as m]))

; The matrix.core library is slow at operations on large matrices. For example,
; back prop is over 20 times slower when performed with core.matrix compared to
; EJML. So core.matrix matrices are converted to SimpleMatrix objects for
; training the network, then at the end, the weights are converted back to
; core.matrix for testing.

; Training set features and labels
(def X
  (to-simple-matrix
    (m/matrix (extract-features (read-csv "resources/sample-train.csv")))))

(def y (m/matrix (extract-labels (read-csv "resources/sample-train.csv"))))
(def Y (to-simple-matrix (class-vector y)))

; Numbers of units in each layer
(def input-layer (.numCols X))
(def hidden-layer 49)
(def output-layer 10)

; Randomised network weights
(def theta1
  (to-simple-matrix
    (init-weights input-layer hidden-layer)))

(def theta2
  (to-simple-matrix
    (init-weights hidden-layer output-layer)))

; Hyperparams
(def lambda 1)

; Cost function for training set
(def cf-train (cost-function X Y lambda))

(defn descend
  "Perform a descent step by subtracting gradient from weights."
  [weights gradients alpha]
  (let [[theta1 theta2] weights
        [theta1-grad theta2-grad] gradients]
    [(.minus theta1 (.scale theta1-grad alpha))
     (.minus theta2 (.scale theta2-grad alpha))]))

(defn minimiser
  "Finds minimum of a cost function `cf` from given starting weights.
  Function `cf` must return both a value and a vector of partial derivatives.
  (This minimiser is a naive gradient descent at the moment.)"
  [cf start iters]
  (let [alpha 1.0]
    (reduce
      (fn [weights i]
        (let [[J gradients] (cf weights)]
          (println "Iteration" i "cost:" J)
          (descend weights gradients alpha)))
      start
      (range iters))))

(defn train
  "Trains network and returns weights in core.matrix."
  []
  (let [weights (minimiser cf-train [theta1 theta2] 50)
        [theta1 theta2] weights
        theta1-core (to-core-matrix theta1)
        theta2-core (to-core-matrix theta2)]
    [theta1-core theta2-core]))

