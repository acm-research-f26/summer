using UnityEngine;

public class ExitEscape : MonoBehaviour
{
    SpriteRenderer renderer;
    public Sprite closedSprite;
    public bool activated;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        GameManagerScript.lockdownInitiated += InLockdown;
        activated = true;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void InLockdown()
    {
        activated = false;
        renderer.sprite = closedSprite;
    }
}
