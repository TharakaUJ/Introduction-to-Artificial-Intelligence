(define (problem move-problem-d)
  (:domain move-domain)

  (:objects
    A B C D - room
  )

  (:init
    (connected A B)
    (connected B A)
    (connected B C)
    (connected C B)
    (connected C D)
    (connected D A)
    (at A)
  )

  (:goal
    (at D)
  )
)

