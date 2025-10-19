(define (domain wumpus-simple)
  (:requirements :strips :typing)
  
  (:types
    location - object
  )
  
  (:predicates
    (at ?loc - location)              ; agent is at location
    (adjacent ?loc1 ?loc2 - location) ; locations are adjacent
    (has-gold)                        ; agent has the gold
    (gold-at ?loc - location)         ; gold is at location
    (pit-at ?loc - location)          ; pit is at location
    (wumpus-at ?loc - location)       ; wumpus is at location
    (wumpus-dead)                     ; wumpus is dead
    (has-arrow)                       ; agent has arrow
    (safe ?loc - location)            ; location is safe to enter
  )
  
  (:action move
    :parameters (?from ?to - location)
    :precondition (and 
      (at ?from)
      (adjacent ?from ?to)
      (safe ?to)
    )
    :effect (and 
      (not (at ?from))
      (at ?to)
    )
  )
  
  (:action pick-gold
    :parameters (?loc - location)
    :precondition (and 
      (at ?loc)
      (gold-at ?loc)
    )
    :effect (and 
      (has-gold)
      (not (gold-at ?loc))
    )
  )
)
