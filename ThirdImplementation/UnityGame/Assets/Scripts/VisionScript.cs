using UnityEngine;

public class VisionScript : MonoBehaviour
{
    GuardScript guard;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        guard = transform.parent.gameObject.GetComponent<GuardScript>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter2D(Collider2D collision)
    {
        if(collision.gameObject.name == "Player")
        {
            guard.OnSuspiciousSighting(collision.transform.position);
        }
        else if(collision.gameObject.name == "diamond")
        {
            guard.OnBrokenDiamond(collision.transform.position);
        }
    }
}
