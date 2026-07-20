using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerScript : MonoBehaviour
{
    private Rigidbody2D rb;
    public BoxCollider2D tungCollider;
    BoxCollider2D plrCollider;

    public bool isJumping;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        plrCollider = GetComponent<BoxCollider2D>();
        isJumping = false;

        Physics2D.IgnoreCollision(plrCollider, tungCollider);
    }

    // Update is called once per frame
    void Update()
    {
        if(Keyboard.current.aKey.isPressed)
        {
            rb.linearVelocityX = -10;
        }
        else if(Keyboard.current.dKey.isPressed)
        {
            rb.linearVelocityX = 10;
        }

        if((Keyboard.current.spaceKey.isPressed || Keyboard.current.wKey.isPressed) && !isJumping)
        {
            isJumping = true;
            rb.linearVelocityY = 50;
        }
    }

    void OnCollisionEnter2D(Collision2D collision)
    {
        foreach (ContactPoint2D contact in collision.contacts)
        {
            if (contact.normal.y > 0.7f) isJumping = false;
        }
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "exit-door" && collision.gameObject.GetComponent<ExitEscape>().activated && GameManagerScript.instance.diamondStolen )
        {
            GameManagerScript.instance.WinGame();
        }
        else if(collision.gameObject.name == "the-hole" && collision.gameObject.GetComponent<HoleEscape>().activated && GameManagerScript.instance.diamondStolen )
        {
            GameManagerScript.instance.WinGame();
        }
        else if(collision.gameObject.name == "sahur" && GameManagerScript.instance.inLockdown)
        {
            GameManagerScript.instance.LoseGame();
        }

        
    }
}