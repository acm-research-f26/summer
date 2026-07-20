using UnityEngine;

public enum CurrentAction
{
    MoveToPlayerOrLastPointSpotted = 1,
    RaiseAlarm = 2,
    InvestigateSound = 3,
    WanderToRandomPlace = 4
};

public class GuardScript : MonoBehaviour
{
    CurrentAction currentBehavior;
    public Vector2 lastPlayerPointSpotted;
    Vector2 targetPosition;
    Vector2 lastSound;
    float movementSpeed;
    public AudioClip angryMusic;
    Rigidbody2D rb;
    AudioSource audiosource;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        currentBehavior = CurrentAction.WanderToRandomPlace;
        lastPlayerPointSpotted = transform.position;
        targetPosition = new Vector2(Random.Range(-73f, 180f), transform.position.y);
        lastSound = transform.position;
        audiosource = GetComponent<AudioSource>();

        movementSpeed = 5;

        rb = GetComponent<Rigidbody2D>();

        GameManagerScript.lockdownInitiated += OnLockdown;

        GameManagerScript.soundOccurred += OnSound;
    }

    CurrentAction PickNewAction()
    {
        CurrentAction chosenAction = (CurrentAction) Random.Range(1, 5);
        if(chosenAction == CurrentAction.WanderToRandomPlace)
        {
            targetPosition = new Vector2(Random.Range(-73f, 180f), transform.position.y);
        }

        Debug.Log($"chosen action is {chosenAction}");

        return chosenAction;

    }

    void DoMoveAction()
    {
        Vector2 movingPosition = Vector2.MoveTowards(transform.position, new Vector2(targetPosition.x, transform.position.y), movementSpeed * Time.deltaTime);
        rb.MovePosition(movingPosition);
        if(Vector2.Distance(transform.position, targetPosition) < 1f)
        {
            currentBehavior = PickNewAction();
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        switch (currentBehavior)
        {
            case CurrentAction.MoveToPlayerOrLastPointSpotted:
                targetPosition = lastPlayerPointSpotted;
                DoMoveAction();
                break;
            case CurrentAction.RaiseAlarm:
                GameManagerScript.instance.RaiseAlarm();
                currentBehavior = PickNewAction();
                break;
            case CurrentAction.InvestigateSound:
                targetPosition = lastSound;
                DoMoveAction();
                break;
            case CurrentAction.WanderToRandomPlace:
                DoMoveAction();
                break;
            default:
                break;
        }

        float direction = targetPosition.x - transform.position.x;
        if(direction < 0)
        {
            transform.localScale = new Vector2(1, transform.localScale.y);
        }
        else if(direction > 0)
        {
            transform.localScale = new Vector2(-1, transform.localScale.y);
        }
    }

    void OnLockdown()
    {
        movementSpeed *= 2;
        audiosource.Stop();
        audiosource.clip = angryMusic;
        audiosource.volume = 0.5f;
        audiosource.Play();
    }

    void OnSound(Vector2 location)
    {
        if(Vector2.Distance(transform.position, location) < 50f)
        {
            lastSound = location;
        }
    }
}
