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
            guard.OnPlayerSighting(collision.transform.position);
        }
        else if(collision.gameObject.name == "diamond" && collision.GetComponent<DiamondScript>().isBroken)
        {
            guard.OnBrokenDiamond(collision.transform.position);
        }
        else if (collision.gameObject.name.Contains("vase"))
        {
            guard.OnVaseSpotted(collision.gameObject.GetComponent<VaseScript>());
        }
    }
}
