using UnityEngine;

public class HoleEscape : MonoBehaviour
{
    SpriteRenderer renderer;
    public bool activated;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        renderer = GetComponent<SpriteRenderer>();
        GameManagerScript.lockdownInitiated += InLockdown;
        activated = false;
        renderer.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void InLockdown()
    {
        activated = true;
        renderer.enabled = true;
    }
}
