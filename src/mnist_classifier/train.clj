(ns mnist-classifier.train
  (:use [mnist-classifier.util]
        [mnist-classifier.cost])
  (:require [clojure.core.matrix :as m]))

; Training set features and labels
(def X (m/matrix (extract-features (read-csv "resources/sample-train.csv"))))
(def y (m/matrix (extract-labels (read-csv "resources/sample-train.csv"))))

; Numbers of units in each layer
(def input-layer (count (first X)))
(def hidden-layer 49)
(def output-layer 10)

; Randomised network weights
(def theta1 (init-weights input-layer hidden-layer))
(def theta2 (init-weights hidden-layer output-layer))

; Hyperparams
(def lambda 1)

; Cost function for training set
(def cf-train (cost-function X y lambda))

(defn descend
  "Perform a descent step by subtracting gradient from weights."
  [weights gradients]
  (let [[theta1 theta2] weights
        [theta1-grad theta2-grad] gradients]
    [(m/sub theta1 theta1-grad) (m/sub theta2 theta2-grad)]))

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
          (descend weights (m/emul alpha gradients))))
      start
      (range iters))))

(defn train
  "Trains network and returns weights."
  []
  (minimiser cf-train [theta1 theta2] 3))

