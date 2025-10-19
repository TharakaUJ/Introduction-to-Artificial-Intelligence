(define (problem wumpus-simple-problem)
  (:domain wumpus-simple)
  
  (:objects
    loc-1-1 loc-1-2 loc-1-3 loc-1-4
    loc-2-1 loc-2-2 loc-2-3 loc-2-4
    loc-3-1 loc-3-2 loc-3-3 loc-3-4
    loc-4-1 loc-4-2 loc-4-3 loc-4-4 - location
  )
  
  (:init
    ; Agent starts at (1,1)
    (at loc-1-3)
    
    ; Gold location
    (gold-at loc-4-2)
    
    ; Wumpus location
    (wumpus-at loc-1-1)
    
    ; Pit locations
    (pit-at loc-3-1)
    (pit-at loc-3-2)
    (pit-at loc-2-4)
    
    ; Define adjacency relationships for 4x4 grid
    ; Row 1 adjacencies
    (adjacent loc-1-1 loc-1-2) (adjacent loc-1-2 loc-1-1)
    (adjacent loc-1-2 loc-1-3) (adjacent loc-1-3 loc-1-2)
    (adjacent loc-1-3 loc-1-4) (adjacent loc-1-4 loc-1-3)
    (adjacent loc-1-1 loc-2-1) (adjacent loc-2-1 loc-1-1)
    (adjacent loc-1-2 loc-2-2) (adjacent loc-2-2 loc-1-2)
    (adjacent loc-1-3 loc-2-3) (adjacent loc-2-3 loc-1-3)
    (adjacent loc-1-4 loc-2-4) (adjacent loc-2-4 loc-1-4)
    
    ; Row 2 adjacencies
    (adjacent loc-2-1 loc-2-2) (adjacent loc-2-2 loc-2-1)
    (adjacent loc-2-2 loc-2-3) (adjacent loc-2-3 loc-2-2)
    (adjacent loc-2-3 loc-2-4) (adjacent loc-2-4 loc-2-3)
    (adjacent loc-2-1 loc-3-1) (adjacent loc-3-1 loc-2-1)
    (adjacent loc-2-2 loc-3-2) (adjacent loc-3-2 loc-2-2)
    (adjacent loc-2-3 loc-3-3) (adjacent loc-3-3 loc-2-3)
    (adjacent loc-2-4 loc-3-4) (adjacent loc-3-4 loc-2-4)
    
    ; Row 3 adjacencies
    (adjacent loc-3-1 loc-3-2) (adjacent loc-3-2 loc-3-1)
    (adjacent loc-3-2 loc-3-3) (adjacent loc-3-3 loc-3-2)
    (adjacent loc-3-3 loc-3-4) (adjacent loc-3-4 loc-3-3)
    (adjacent loc-3-1 loc-4-1) (adjacent loc-4-1 loc-3-1)
    (adjacent loc-3-2 loc-4-2) (adjacent loc-4-2 loc-3-2)
    (adjacent loc-3-3 loc-4-3) (adjacent loc-4-3 loc-3-3)
    (adjacent loc-3-4 loc-4-4) (adjacent loc-4-4 loc-3-4)
    
    ; Row 4 adjacencies
    (adjacent loc-4-1 loc-4-2) (adjacent loc-4-2 loc-4-1)
    (adjacent loc-4-2 loc-4-3) (adjacent loc-4-3 loc-4-2)
    (adjacent loc-4-3 loc-4-4) (adjacent loc-4-4 loc-4-3)
    
    ; Safe locations (all locations except pits and wumpus location initially)
    (safe loc-1-1) (safe loc-1-2) (safe loc-1-4)
    (safe loc-2-1) (safe loc-2-2) (safe loc-2-3)
    (safe loc-3-3) (safe loc-3-4)
    (safe loc-4-1) (safe loc-4-2) (safe loc-4-3) (safe loc-4-4)
  )
  
  (:goal
    (and 
      (has-gold)
      (at loc-1-1)
    )
  )
)
